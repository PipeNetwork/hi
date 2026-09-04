use crate::Agent;

impl Agent {
    /// Cloneable inventory used by a host durability monitor to stop live
    /// foreground commands when workspace authority is lost.
    pub fn foreground_process_registry(&self) -> hi_tools::ForegroundProcessRegistry {
        self.runtime.process_runner().foreground_registry()
    }
}
