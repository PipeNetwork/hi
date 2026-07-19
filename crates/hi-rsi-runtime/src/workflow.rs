use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct StageId(pub String);

impl From<&str> for StageId {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    DeterministicTransform,
    ModelInvocation,
    ToolInvocation,
    ParallelFanOut,
    Aggregation,
    PolicyGate,
    VerificationGate,
    HumanApprovalGate,
    TerminalSuccess,
    TerminalFailure,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageDefinition {
    pub kind: StageKind,
    #[serde(default)]
    pub model_role: Option<String>,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub iteration_limit: Option<u32>,
    #[serde(default)]
    pub trusted: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionCondition {
    Always,
    StagePassed,
    StageFailed,
    BudgetRemaining,
    HumanApproved,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionRule {
    pub from: StageId,
    pub to: StageId,
    pub condition: TransitionCondition,
    pub priority: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowLimits {
    pub maximum_transitions: u32,
    pub maximum_parallelism: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowGraph {
    pub entry: StageId,
    pub stages: BTreeMap<StageId, StageDefinition>,
    pub edges: Vec<TransitionRule>,
    pub limits: WorkflowLimits,
}

impl WorkflowGraph {
    pub fn validate(
        &self,
        authorized_roles: &BTreeSet<String>,
        authorized_tools: &BTreeSet<String>,
    ) -> Result<()> {
        ensure!(!self.stages.is_empty(), "workflow has no stages");
        ensure!(
            self.stages.contains_key(&self.entry),
            "workflow entry stage is missing"
        );
        ensure!(
            self.limits.maximum_transitions > 0,
            "workflow transition limit must be positive"
        );
        ensure!(
            self.limits.maximum_parallelism > 0,
            "workflow parallelism limit must be positive"
        );
        for (id, stage) in &self.stages {
            validate_id(id)?;
            if stage.kind == StageKind::ModelInvocation {
                let role = stage
                    .model_role
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("model stage {} has no role", id.0))?;
                ensure!(
                    authorized_roles.contains(role),
                    "model stage {} uses unauthorized role {role}",
                    id.0
                );
            } else {
                ensure!(
                    stage.model_role.is_none(),
                    "non-model stage {} declares a model role",
                    id.0
                );
            }
            if stage.kind == StageKind::ToolInvocation {
                let tool = stage
                    .tool
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("tool stage {} has no tool", id.0))?;
                ensure!(
                    authorized_tools.contains(tool),
                    "tool stage {} uses unauthorized tool {tool}",
                    id.0
                );
            } else {
                ensure!(
                    stage.tool.is_none(),
                    "non-tool stage {} declares a tool",
                    id.0
                );
            }
            if stage.kind == StageKind::VerificationGate {
                ensure!(stage.trusted, "verification gate {} is not trusted", id.0);
            }
        }
        let mut priorities = BTreeSet::new();
        for edge in &self.edges {
            ensure!(
                self.stages.contains_key(&edge.from) && self.stages.contains_key(&edge.to),
                "workflow edge references a missing stage"
            );
            ensure!(
                priorities.insert((edge.from.clone(), edge.priority)),
                "workflow has duplicate transition priority"
            );
        }
        self.validate_reachability()?;
        self.validate_cycles()?;
        self.validate_success_requires_verification()
    }

    fn validate_reachability(&self) -> Result<()> {
        let mut reachable = BTreeSet::new();
        let mut queue = VecDeque::from([self.entry.clone()]);
        while let Some(stage) = queue.pop_front() {
            if !reachable.insert(stage.clone()) {
                continue;
            }
            queue.extend(
                self.edges
                    .iter()
                    .filter(|edge| edge.from == stage)
                    .map(|edge| edge.to.clone()),
            );
        }
        ensure!(
            reachable.len() == self.stages.len(),
            "workflow contains unreachable stages"
        );
        ensure!(
            reachable.iter().any(|id| {
                matches!(
                    self.stages[id].kind,
                    StageKind::TerminalSuccess | StageKind::TerminalFailure
                )
            }),
            "workflow has no reachable terminal stage"
        );
        Ok(())
    }

    fn validate_cycles(&self) -> Result<()> {
        fn visit(
            graph: &WorkflowGraph,
            node: &StageId,
            active: &mut BTreeSet<StageId>,
            finished: &mut BTreeSet<StageId>,
        ) -> Result<()> {
            if finished.contains(node) {
                return Ok(());
            }
            active.insert(node.clone());
            for next in graph
                .edges
                .iter()
                .filter(|edge| &edge.from == node)
                .map(|edge| &edge.to)
            {
                if active.contains(next) {
                    let current_limited = graph.stages[node].iteration_limit.is_some();
                    let next_limited = graph.stages[next].iteration_limit.is_some();
                    ensure!(
                        current_limited || next_limited,
                        "workflow cycle has no explicit iteration limit"
                    );
                } else {
                    visit(graph, next, active, finished)?;
                }
            }
            active.remove(node);
            finished.insert(node.clone());
            Ok(())
        }
        visit(
            self,
            &self.entry,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
        )
    }

    fn validate_success_requires_verification(&self) -> Result<()> {
        let mut queue = VecDeque::from([(self.entry.clone(), false)]);
        let mut seen = BTreeSet::new();
        while let Some((id, verified)) = queue.pop_front() {
            let stage = &self.stages[&id];
            let verified = verified || stage.kind == StageKind::VerificationGate;
            if !seen.insert((id.clone(), verified)) {
                continue;
            }
            if stage.kind == StageKind::TerminalSuccess && !verified {
                bail!("terminal success can bypass trusted verification");
            }
            for next in self
                .edges
                .iter()
                .filter(|edge| edge.from == id)
                .map(|edge| edge.to.clone())
            {
                queue.push_back((next, verified));
            }
        }
        Ok(())
    }

    pub fn default_coding() -> Self {
        use StageKind::*;
        let ordered = [
            ("intake", DeterministicTransform, false),
            ("normalize_requirements", ModelInvocation, false),
            ("explore_repository", ModelInvocation, false),
            ("diagnose", ModelInvocation, false),
            ("plan", ModelInvocation, false),
            ("implement", ModelInvocation, false),
            ("compile", ToolInvocation, false),
            ("test", ToolInvocation, false),
            ("review", ModelInvocation, false),
            ("verify", VerificationGate, true),
            ("complete", TerminalSuccess, false),
        ];
        let mut stages = BTreeMap::new();
        for (name, kind, trusted) in ordered {
            let role = match name {
                "normalize_requirements" => Some("requirement_normalizer"),
                "explore_repository" => Some("repository_explorer"),
                "diagnose" => Some("diagnostician"),
                "plan" => Some("planner"),
                "implement" => Some("implementer"),
                "review" => Some("reviewer"),
                _ => None,
            };
            let tool = match name {
                "compile" => Some("cargo_check"),
                "test" => Some("cargo_test"),
                _ => None,
            };
            stages.insert(
                StageId::from(name),
                StageDefinition {
                    kind,
                    model_role: role.map(str::to_owned),
                    tool: tool.map(str::to_owned),
                    iteration_limit: None,
                    trusted,
                },
            );
        }
        let ids = ordered.map(|(name, _, _)| StageId::from(name));
        let edges = ids
            .windows(2)
            .enumerate()
            .map(|(index, pair)| TransitionRule {
                from: pair[0].clone(),
                to: pair[1].clone(),
                condition: TransitionCondition::StagePassed,
                priority: index as u16,
            })
            .collect();
        Self {
            entry: StageId::from("intake"),
            stages,
            edges,
            limits: WorkflowLimits {
                maximum_transitions: 100,
                maximum_parallelism: 4,
            },
        }
    }
}

fn validate_id(id: &StageId) -> Result<()> {
    ensure!(
        !id.0.is_empty()
            && id.0.len() <= 128
            && id
                .0
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
        "invalid workflow stage id"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authorities() -> (BTreeSet<String>, BTreeSet<String>) {
        (
            [
                "requirement_normalizer",
                "repository_explorer",
                "diagnostician",
                "planner",
                "implementer",
                "reviewer",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            ["cargo_check", "cargo_test"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        )
    }

    #[test]
    fn default_graph_is_valid_and_requires_trusted_verification() {
        let (roles, tools) = authorities();
        WorkflowGraph::default_coding()
            .validate(&roles, &tools)
            .unwrap();
    }

    #[test]
    fn direct_success_bypass_is_rejected() {
        let (roles, tools) = authorities();
        let mut graph = WorkflowGraph::default_coding();
        graph.edges.push(TransitionRule {
            from: StageId::from("intake"),
            to: StageId::from("complete"),
            condition: TransitionCondition::Always,
            priority: 99,
        });
        assert!(graph.validate(&roles, &tools).is_err());
    }

    #[test]
    fn unbounded_cycle_and_unauthorized_role_are_rejected() {
        let (roles, tools) = authorities();
        let mut graph = WorkflowGraph::default_coding();
        graph.edges.push(TransitionRule {
            from: StageId::from("review"),
            to: StageId::from("implement"),
            condition: TransitionCondition::StageFailed,
            priority: 99,
        });
        assert!(graph.validate(&roles, &tools).is_err());
        graph
            .stages
            .get_mut(&StageId::from("review"))
            .unwrap()
            .iteration_limit = Some(2);
        graph
            .stages
            .get_mut(&StageId::from("implement"))
            .unwrap()
            .model_role = Some("unauthorized".into());
        assert!(graph.validate(&roles, &tools).is_err());
    }
}
