use std::sync::{Arc, Mutex};

use super::*;

#[derive(Default)]
struct DenyLiveWriterLifecycle {
    registrations: Mutex<Vec<crate::BackgroundJobRegistration>>,
}

#[async_trait::async_trait]
impl crate::BackgroundJobLifecycle for DenyLiveWriterLifecycle {
    async fn register(&self, registration: crate::BackgroundJobRegistration) -> Result<(), String> {
        let effect = registration.effect;
        self.registrations.lock().unwrap().push(registration);
        if effect == crate::BackgroundJobEffect::LiveWriter {
            Err("PipeFS live writers are unavailable".into())
        } else {
            Ok(())
        }
    }

    async fn observe_terminal(
        &self,
        _id: &crate::BackgroundJobId,
        _terminal: crate::BackgroundJobTerminal,
        _detail: Option<String>,
    ) -> Result<crate::BackgroundJobPublication, String> {
        panic!("a denied download must never start or reach terminal settlement")
    }

    async fn pending(&self, _source_id: &str) -> Vec<crate::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(
        &self,
        _pending: &[crate::BackgroundJobId],
    ) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn web_download_live_writer_denial_happens_before_process_start() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("denied-download.bin");
    let background = crate::BackgroundRegistry::default();
    let lifecycle = Arc::new(DenyLiveWriterLifecycle::default());
    background.set_job_lifecycle(lifecycle.clone());
    let command = format!(
        "printf started > {}",
        shell_quote(&output.to_string_lossy())
    );

    let result = spawn_download_process(directory.path(), &background, &command).await;

    let error = result.expect_err("PipeFS-like lifecycle denial must fail the download");
    assert!(
        error.to_string().contains("live writers are unavailable"),
        "unexpected denial: {error:#}"
    );
    assert!(!output.exists(), "denied download created an output file");
    assert!(
        background.ids().is_empty(),
        "denied download retained a background process"
    );
    let registrations = lifecycle.registrations.lock().unwrap();
    assert_eq!(registrations.len(), 1);
    assert_eq!(registrations[0].kind, crate::BackgroundJobKind::Process);
    assert_eq!(
        registrations[0].effect,
        crate::BackgroundJobEffect::LiveWriter
    );
}

#[tokio::test]
async fn web_download_empty_source_rejected() {
    let out = run_web_download(r#"{"source":""}"#).await;
    assert!(out.is_err());
}

#[tokio::test]
async fn web_download_full_url_resolves_directly() {
    let (target, name) = resolve_download("https://example.com/file.gguf", None)
        .await
        .unwrap();
    assert!(matches!(
        target,
        DownloadTarget::Url(ref url) if url == "https://example.com/file.gguf"
    ));
    assert_eq!(name, "file.gguf");
}
