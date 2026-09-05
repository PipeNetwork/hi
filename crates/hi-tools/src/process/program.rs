use super::execution::{OutputMode, capture_child_with_output_mode};
use super::*;

impl ProcessRunner {
    /// Run an executable directly, keeping filesystem-derived arguments out of
    /// a shell parser. This is used for filename-sensitive syntax checks and
    /// other internal commands.
    pub async fn run_program<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Duration,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.run_program_maybe_timeout(program, args, Some(timeout))
            .await
    }

    /// Run an executable directly with an optional outer deadline.
    ///
    /// `None` leaves productive work active until the program exits or the
    /// returned future is cancelled. Cancellation still drops the process
    /// group guard and removes the direct child and all of its descendants.
    pub async fn run_program_maybe_timeout<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Option<Duration>,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.run_program_with_output_mode(program, args, timeout, OutputMode::Diagnostics)
            .await
    }

    /// Search results are source evidence, even when they quote test output.
    /// Apply the normal byte/character limits without diagnostic filtering.
    pub(crate) async fn run_program_plain_maybe_timeout<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Option<Duration>,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.run_program_with_output_mode(program, args, timeout, OutputMode::Plain)
            .await
    }

    async fn run_program_with_output_mode<I, S>(
        &self,
        program: impl AsRef<OsStr>,
        args: I,
        timeout: Option<Duration>,
        output_mode: OutputMode,
    ) -> Result<ProcessExecution>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let started = Instant::now();
        let (wrapped_program, wrapped_args) =
            self.sandbox
                .wrap_program_in(program.as_ref(), args, &self.root);
        let mut command = Command::new(wrapped_program);
        command.args(wrapped_args);
        self.configure(&mut command);
        if self.sandbox.is_enforced() {
            command.env(crate::sandbox::NESTED_SANDBOX_ENV, "1");
        }
        let child = command.spawn().context("failed to spawn program")?;
        capture_child_with_output_mode(
            child,
            timeout,
            &mut |_| {},
            started,
            &self.foreground,
            output_mode,
        )
        .await
    }
}
