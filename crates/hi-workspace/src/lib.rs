//! Workspace coordination contracts shared by the agent and storage backends.
//!
//! This crate deliberately contains no filesystem, database, or transport
//! implementation. Hosts select a local or remote backend while callers use
//! the same admission, settlement, job, barrier, and recovery protocol.

mod controller;
mod failpoint;
mod harness_settings;
mod in_memory;
mod job_output;
mod job_registry;
mod model;
mod recovery;
mod resource;
mod settings;
mod tool_diagnostic;
mod verified_candidate;

pub use controller::{
    MutationPermit, PermitAbandonment, PermitClaimError, PermitIssuer, WorkspaceController,
};
pub use failpoint::*;
pub use harness_settings::*;
pub use in_memory::InMemoryWorkspaceController;
pub use job_output::*;
pub use job_registry::*;
pub use model::*;
pub use recovery::*;
pub use resource::*;
pub use settings::*;
pub use tool_diagnostic::*;
pub use verified_candidate::*;

/// Schema version for serialized workspace coordination records.
pub const WORKSPACE_CONTRACT_SCHEMA_VERSION: u16 = 1;

#[cfg(test)]
mod job_registry_tests;
