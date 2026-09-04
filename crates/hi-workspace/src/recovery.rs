use crate::{BindingId, JobId, OperationId, RecoveryId};

/// Stable identity for a writer job whose lifecycle crossed a process restart.
pub fn restart_job_recovery_id(binding_id: &BindingId, epoch: u64, job_id: &JobId) -> RecoveryId {
    recovery_id(format!("{binding_id}:{epoch}:{job_id}"))
}

/// Stable identity for an operation whose lifecycle crossed a process restart.
pub fn restart_operation_recovery_id(
    binding_id: &BindingId,
    epoch: u64,
    operation_id: &OperationId,
) -> RecoveryId {
    recovery_id(format!("{binding_id}:{epoch}:operation:{operation_id}"))
}

fn recovery_id(identity: String) -> RecoveryId {
    RecoveryId::new(uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, identity.as_bytes()).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_ids_are_stable_and_kind_separated() {
        let binding = BindingId::new("binding");
        let operation = restart_operation_recovery_id(&binding, 7, &OperationId::new("same"));
        let job = restart_job_recovery_id(&binding, 7, &JobId::new("same"));

        assert_eq!(
            operation,
            restart_operation_recovery_id(&binding, 7, &OperationId::new("same"))
        );
        assert_ne!(operation, job);
    }
}
