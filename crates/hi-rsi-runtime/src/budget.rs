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
}
