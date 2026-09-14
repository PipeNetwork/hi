//! Session knobs that can change while a turn is in flight.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use hi_ai::ReasoningEffort;

use crate::command::EffortArg;
use crate::ui::PermissionMode;

/// Cloneable handle for permission, reasoning, and model. The turn loop reads
/// these on each model round and tool call so `/yolo` and `/effort` apply
/// without waiting for the turn to finish.
#[derive(Clone)]
pub struct LiveSettings {
    permission: Arc<AtomicU8>,
    reasoning: Arc<Mutex<Option<ReasoningEffort>>>,
    model: Arc<Mutex<String>>,
}

impl LiveSettings {
    pub fn new(
        model: String,
        permission: PermissionMode,
        reasoning: Option<ReasoningEffort>,
    ) -> Self {
        Self {
            permission: Arc::new(AtomicU8::new(permission.as_u8())),
            reasoning: Arc::new(Mutex::new(reasoning)),
            model: Arc::new(Mutex::new(model)),
        }
    }

    pub fn permission_mode(&self) -> PermissionMode {
        PermissionMode::from_u8(self.permission.load(Ordering::Acquire))
    }

    pub fn set_permission_mode(&self, mode: PermissionMode) {
        self.permission.store(mode.as_u8(), Ordering::Release);
    }

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        lock(&self.reasoning)
    }

    pub fn set_reasoning_effort(&self, effort: Option<ReasoningEffort>) {
        *lock_mut(&self.reasoning) = effort;
    }

    pub fn apply_effort_arg(&self, effort: EffortArg) {
        self.set_reasoning_effort(match effort {
            EffortArg::Off => None,
            EffortArg::Level(level) => Some(level),
        });
    }

    pub fn model(&self) -> String {
        lock(&self.model)
    }

    pub fn set_model(&self, model: String) {
        *lock_mut(&self.model) = model;
    }
}

fn lock<T: Clone>(mutex: &Mutex<T>) -> T {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn lock_mut<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
