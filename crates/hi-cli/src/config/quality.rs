use super::*;

/// Resolved project-quality settings. Precedence is CLI, `.hi/config.toml`,
/// then built-in automatic detection/defaults.
#[derive(Clone, Debug)]
pub struct QualitySettings {
    pub verification: VerificationMode,
    pub max_verify_repairs: u32,
    pub review: ReviewPolicy,
    pub clippy: bool,
    pub lsp_mode: LspMode,
    pub tool_set: ToolSet,
    pub context_exclusions: Vec<String>,
    pub race: RaceSettings,
}

#[derive(Clone, Debug, Default)]
pub struct RaceSettings {
    pub enabled: bool,
    pub max_candidates: u32,
    pub max_concurrency: usize,
    pub targets: Vec<hi_race::RaceTarget>,
    pub fuzz: Option<hi_race::FuzzConfig>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProjectConfig {
    #[serde(default)]
    quality: ProjectQuality,
    #[serde(default)]
    race: ProjectRace,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProjectRace {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    max_candidates: Option<u32>,
    #[serde(default)]
    max_concurrency: Option<usize>,
    #[serde(default)]
    fuzz_command: Option<String>,
    #[serde(default)]
    fuzz_timeout_secs: Option<u64>,
    #[serde(default)]
    targets: Vec<hi_race::RaceTarget>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ProjectVerificationMode {
    Auto,
    Explicit,
    Disabled,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProjectQuality {
    #[serde(default, alias = "verification_mode")]
    verification: Option<ProjectVerificationMode>,
    /// Ordered commands used by `verification = "explicit"`. `verify` is an
    /// accepted alias for early 0.2 preview files.
    #[serde(default, alias = "verify")]
    stages: Vec<String>,
    #[serde(default)]
    max_verify_repairs: Option<u32>,
    #[serde(default)]
    review: Option<ReviewPolicy>,
    #[serde(default)]
    clippy: Option<bool>,
    #[serde(default, alias = "lsp_mode")]
    lsp: Option<LspMode>,
    #[serde(default)]
    tool_set: Option<ToolSet>,
    #[serde(default)]
    context_exclusions: Vec<String>,
}

/// Load and resolve `.hi/config.toml` quality policy for `root`.
pub fn resolve_quality(cli: &Cli, root: &Path) -> Result<QualitySettings> {
    let path = root.join(".hi/config.toml");
    let project = if path.exists() {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading project config {}", path.display()))?;
        toml::from_str::<ProjectConfig>(&text)
            .with_context(|| format!("parsing project config {}", path.display()))?
    } else {
        ProjectConfig::default()
    };
    let quality = project.quality;
    let project_race = project.race;

    let project_verification = match quality.verification {
        Some(ProjectVerificationMode::Disabled) => {
            if !quality.stages.is_empty() {
                bail!("[quality] cannot combine verification = \"disabled\" with stages");
            }
            VerificationMode::Disabled
        }
        Some(ProjectVerificationMode::Explicit) => {
            if quality.stages.is_empty() {
                bail!("[quality] verification = \"explicit\" requires at least one stage");
            }
            VerificationMode::Explicit(quality_stages(&quality.stages)?)
        }
        Some(ProjectVerificationMode::Auto) => {
            if !quality.stages.is_empty() {
                bail!("[quality] cannot combine verification = \"auto\" with stages");
            }
            VerificationMode::Auto
        }
        None if !quality.stages.is_empty() => {
            VerificationMode::Explicit(quality_stages(&quality.stages)?)
        }
        None => VerificationMode::Auto,
    };

    let mut verification = if cli.no_verify {
        VerificationMode::Disabled
    } else if !cli.verify.is_empty() {
        VerificationMode::Explicit(quality_stages(&cli.verify)?)
    } else {
        project_verification
    };
    let clippy = cli.clippy || quality.clippy.unwrap_or(false);
    if clippy && matches!(verification, VerificationMode::Auto) {
        let stages = hi_agent::detect_verify_pipeline_with(root, true);
        if !stages.is_empty() {
            verification = VerificationMode::Explicit(stages);
        }
    }

    Ok(QualitySettings {
        verification,
        max_verify_repairs: cli
            .max_verify_repairs
            .or(quality.max_verify_repairs)
            .unwrap_or(2),
        review: cli
            .review
            .map(ReviewPolicy::from)
            .or(quality.review)
            .unwrap_or_default(),
        clippy,
        lsp_mode: cli
            .lsp
            .map(LspMode::from)
            .or(quality.lsp)
            .unwrap_or_default(),
        tool_set: cli
            .tool_set
            .map(ToolSet::from)
            .or(quality.tool_set)
            .unwrap_or_default(),
        context_exclusions: quality.context_exclusions,
        race: RaceSettings {
            enabled: project_race
                .enabled
                .unwrap_or(!project_race.targets.is_empty()),
            max_candidates: project_race
                .max_candidates
                .unwrap_or(hi_race::DEFAULT_MAX_CANDIDATES),
            max_concurrency: project_race.max_concurrency.unwrap_or(2),
            targets: project_race.targets,
            fuzz: project_race
                .fuzz_command
                .map(|command| hi_race::FuzzConfig {
                    command,
                    timeout_secs: project_race
                        .fuzz_timeout_secs
                        .unwrap_or(hi_race::DEFAULT_FUZZ_TIMEOUT_SECS),
                }),
        },
    })
}

pub(crate) fn quality_stages(commands: &[String]) -> Result<Vec<VerifyStage>> {
    if let Some((index, _)) = commands
        .iter()
        .enumerate()
        .find(|(_, command)| command.trim().is_empty())
    {
        bail!("verification stage {} must not be empty", index + 1);
    }
    Ok(commands
        .iter()
        .enumerate()
        .map(|(index, command)| {
            let name = if commands.len() == 1 {
                "verify".to_string()
            } else {
                format!("verify_{}", index + 1)
            };
            VerifyStage::new(name, command.trim().to_string())
        })
        .collect())
}
