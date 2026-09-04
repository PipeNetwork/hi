use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::RuntimeBudgets;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetKind {
    CpuTimeSeconds,
    DiskBytes,
    InputTokens,
    OutputTokens,
    ToolCalls,
    ModelCalls,
    CostMicrousd,
    RepairIterations,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BudgetUsage {
    pub consumed: BTreeMap<BudgetKind, u64>,
    pub reserved: BTreeMap<BudgetKind, u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetReservation {
    id: u64,
    kind: BudgetKind,
    amount: u64,
}

#[derive(Clone, Debug)]
pub struct BudgetLedger {
    limits: BTreeMap<BudgetKind, u64>,
    usage: BudgetUsage,
    reservations: BTreeMap<u64, BudgetReservation>,
    next_id: u64,
}

#[derive(Clone, Debug)]
pub struct SharedBudgetLedger(Arc<Mutex<BudgetLedger>>);

impl BudgetLedger {
    pub fn new(limits: &RuntimeBudgets) -> Self {
        Self {
            limits: BTreeMap::from([
                (BudgetKind::CpuTimeSeconds, limits.cpu_time_seconds),
                (BudgetKind::DiskBytes, limits.disk_bytes),
                (BudgetKind::InputTokens, limits.input_tokens),
                (BudgetKind::OutputTokens, limits.output_tokens),
                (BudgetKind::ToolCalls, limits.tool_calls),
                (BudgetKind::ModelCalls, u64::from(limits.model_calls)),
                (BudgetKind::CostMicrousd, limits.cost_microusd),
                (
                    BudgetKind::RepairIterations,
                    u64::from(limits.repair_iterations),
                ),
            ]),
            usage: BudgetUsage::default(),
            reservations: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub fn reserve(&mut self, kind: BudgetKind, amount: u64) -> Result<BudgetReservation> {
        ensure!(amount > 0, "budget reservations must be positive");
        let available = self.remaining(kind);
        ensure!(amount <= available, "{kind:?} budget exhausted");
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("budget reservation id overflow"))?;
        let reservation = BudgetReservation {
            id: self.next_id,
            kind,
            amount,
        };
        add(&mut self.usage.reserved, kind, amount)?;
        self.reservations.insert(reservation.id, reservation);
        Ok(reservation)
    }

    /// Settle a reservation using the actual amount consumed. The actual value
    /// may be lower than the conservative reservation but never higher.
    pub fn commit(&mut self, reservation: BudgetReservation, actual: u64) -> Result<()> {
        let held = self
            .reservations
            .get(&reservation.id)
            .copied()
            .ok_or_else(|| anyhow!("unknown or already settled budget reservation"))?;
        ensure!(held == reservation, "budget reservation identity mismatch");
        ensure!(
            actual <= held.amount,
            "budget settlement exceeds reservation"
        );
        let held = self.take(reservation)?;
        subtract(&mut self.usage.reserved, held.kind, held.amount)?;
        add(&mut self.usage.consumed, held.kind, actual)
    }

    pub fn release(&mut self, reservation: BudgetReservation) -> Result<()> {
        let held = self.take(reservation)?;
        subtract(&mut self.usage.reserved, held.kind, held.amount)
    }

    pub fn consume(&mut self, kind: BudgetKind, amount: u64) -> Result<()> {
        let reservation = self.reserve(kind, amount)?;
        self.commit(reservation, amount)
    }

    pub fn remaining(&self, kind: BudgetKind) -> u64 {
        let limit = self.limits.get(&kind).copied().unwrap_or(0);
        let consumed = self.usage.consumed.get(&kind).copied().unwrap_or(0);
        let reserved = self.usage.reserved.get(&kind).copied().unwrap_or(0);
        limit.saturating_sub(consumed.saturating_add(reserved))
    }

    pub fn usage(&self) -> &BudgetUsage {
        &self.usage
    }

    fn merge_consumption_floor(&mut self, floor: &BudgetUsage) -> Result<BudgetUsage> {
        ensure!(
            self.reservations.is_empty() && self.usage.reserved.values().all(|amount| *amount == 0),
            "budget consumption floor cannot merge while reservations are active"
        );
        ensure!(
            floor.reserved.values().all(|amount| *amount == 0),
            "budget consumption floor contains unsettled reservations"
        );
        // Validate the complete floor before mutating any dimension so a bad
        // snapshot cannot partially advance the live ledger.
        for (&kind, &amount) in &floor.consumed {
            let limit = self
                .limits
                .get(&kind)
                .copied()
                .ok_or_else(|| anyhow!("budget floor references unknown {kind:?} budget"))?;
            ensure!(
                amount <= limit,
                "budget floor {kind:?} usage {amount} exceeds limit {limit}"
            );
        }
        for (&kind, &amount) in &floor.consumed {
            let consumed = self.usage.consumed.entry(kind).or_default();
            *consumed = (*consumed).max(amount);
        }
        Ok(self.usage.clone())
    }

    fn take(&mut self, requested: BudgetReservation) -> Result<BudgetReservation> {
        let held = self
            .reservations
            .remove(&requested.id)
            .ok_or_else(|| anyhow!("unknown or already settled budget reservation"))?;
        ensure!(held == requested, "budget reservation identity mismatch");
        Ok(held)
    }
}

impl SharedBudgetLedger {
    pub fn new(limits: &RuntimeBudgets) -> Self {
        Self(Arc::new(Mutex::new(BudgetLedger::new(limits))))
    }

    pub fn reserve(&self, kind: BudgetKind, amount: u64) -> Result<BudgetReservation> {
        self.0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .reserve(kind, amount)
    }

    pub fn commit(&self, reservation: BudgetReservation, actual: u64) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .commit(reservation, actual)
    }

    pub fn release(&self, reservation: BudgetReservation) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .release(reservation)
    }

    pub fn consume(&self, kind: BudgetKind, amount: u64) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        let mut ledger = self
            .0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?;
        ledger.consume(kind, amount)
    }

    pub fn usage(&self) -> Result<BudgetUsage> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .usage()
            .clone())
    }

    /// Raise durable consumption to at least `floor` without ever erasing
    /// usage already incurred by this ledger. This is safe for both a fresh
    /// process and an in-process retry using the same ledger, and is
    /// idempotent when the floor was already merged.
    ///
    /// Neither side may contain live reservations: those represent in-flight
    /// work and must settle before a retry boundary.
    pub fn merge_consumption_floor(&self, floor: &BudgetUsage) -> Result<BudgetUsage> {
        self.0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .merge_consumption_floor(floor)
    }

    /// Remaining unreserved capacity for one budget dimension.
    ///
    /// Keep this query on the shared ledger rather than deriving it from a
    /// [`BudgetUsage`] snapshot: reservations made by concurrent stages must
    /// count against transition admission immediately.
    pub fn remaining(&self, kind: BudgetKind) -> Result<u64> {
        Ok(self
            .0
            .lock()
            .map_err(|_| anyhow!("budget ledger lock poisoned"))?
            .remaining(kind))
    }
}

fn add(map: &mut BTreeMap<BudgetKind, u64>, kind: BudgetKind, amount: u64) -> Result<()> {
    let value = map.entry(kind).or_default();
    *value = value
        .checked_add(amount)
        .ok_or_else(|| anyhow!("budget arithmetic overflow"))?;
    Ok(())
}

fn subtract(map: &mut BTreeMap<BudgetKind, u64>, kind: BudgetKind, amount: u64) -> Result<()> {
    let value = map.entry(kind).or_default();
    *value = value
        .checked_sub(amount)
        .ok_or_else(|| anyhow!("budget arithmetic underflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    fn limits() -> RuntimeBudgets {
        RuntimeBudgets {
            wall_time_seconds: 10,
            cpu_time_seconds: 10,
            memory_bytes: 10,
            disk_bytes: 10,
            input_tokens: 10,
            output_tokens: 10,
            tool_calls: 10,
            cost_microusd: 10,
            model_calls: 10,
            repair_iterations: 10,
            trace_bytes: 10,
        }
    }

    #[test]
    fn reservations_cannot_double_spend_or_double_settle() {
        let mut ledger = BudgetLedger::new(&limits());
        let held = ledger.reserve(BudgetKind::ToolCalls, 7).unwrap();
        assert!(ledger.reserve(BudgetKind::ToolCalls, 4).is_err());
        ledger.commit(held, 6).unwrap();
        assert!(ledger.commit(held, 6).is_err());
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls), 4);
    }

    #[test]
    fn concurrent_reservations_honor_one_authoritative_limit() {
        let ledger = SharedBudgetLedger::new(&limits());
        let mut tasks = Vec::new();
        for _ in 0..20 {
            let ledger = ledger.clone();
            tasks.push(thread::spawn(move || {
                ledger.reserve(BudgetKind::ModelCalls, 1).ok()
            }));
        }
        let reservations = tasks
            .into_iter()
            .filter_map(|task| task.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(reservations.len(), 10);
        for reservation in reservations {
            ledger.commit(reservation, 1).unwrap();
        }
        assert_eq!(
            ledger
                .usage()
                .unwrap()
                .consumed
                .get(&BudgetKind::ModelCalls),
            Some(&10)
        );
    }

    #[test]
    fn shared_remaining_includes_consumed_and_reserved_capacity() {
        let ledger = SharedBudgetLedger::new(&limits());
        ledger.consume(BudgetKind::ToolCalls, 3).unwrap();
        let held = ledger.reserve(BudgetKind::ToolCalls, 4).unwrap();
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls).unwrap(), 3);
        ledger.release(held).unwrap();
        assert_eq!(ledger.remaining(BudgetKind::ToolCalls).unwrap(), 7);
    }

    #[test]
    fn consumption_floor_is_monotonic_idempotent_and_atomic() {
        let ledger = SharedBudgetLedger::new(&limits());
        ledger.consume(BudgetKind::ToolCalls, 3).unwrap();
        let merged = ledger
            .merge_consumption_floor(&BudgetUsage {
                consumed: BTreeMap::from([(BudgetKind::ToolCalls, 2), (BudgetKind::ModelCalls, 4)]),
                reserved: BTreeMap::new(),
            })
            .unwrap();
        assert_eq!(merged.consumed.get(&BudgetKind::ToolCalls), Some(&3));
        assert_eq!(merged.consumed.get(&BudgetKind::ModelCalls), Some(&4));
        assert_eq!(
            ledger.merge_consumption_floor(&merged).unwrap(),
            merged,
            "merging the same durable floor twice must be a no-op"
        );

        let error = ledger
            .merge_consumption_floor(&BudgetUsage {
                consumed: BTreeMap::from([
                    (BudgetKind::ToolCalls, 5),
                    (BudgetKind::ModelCalls, 11),
                ]),
                reserved: BTreeMap::new(),
            })
            .expect_err("an over-limit dimension must reject the complete floor");
        assert!(error.to_string().contains("exceeds limit"));
        assert_eq!(
            ledger.usage().unwrap().consumed.get(&BudgetKind::ToolCalls),
            Some(&3),
            "validation must finish before any dimension is raised"
        );

        assert!(
            ledger
                .merge_consumption_floor(&BudgetUsage {
                    consumed: BTreeMap::new(),
                    reserved: BTreeMap::from([(BudgetKind::ToolCalls, 1)]),
                })
                .is_err(),
            "durable floors must never import in-flight reservations"
        );
    }
}
