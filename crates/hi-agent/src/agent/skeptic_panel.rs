//! Grok-style N-skeptic majority panel for `/goal team`.
//!
//! Production default is 3 independent reviews of the same turn, issued in
//! parallel. Majority of *parsed* verdicts decides; a 1–1 tie with one
//! unavailable fails open to APPROVE. Infra-only panels stay
//! [`SkepticVerdict::Unavailable`].

use std::sync::Arc;
use std::time::Duration;

use hi_ai::{ChatRequest, Provider, StreamEvent, Usage};

use super::skeptic::SkepticVerdict;

pub(crate) struct ReviewAttempt {
    pub verdict: SkepticVerdict,
    pub usage: Option<Usage>,
    pub error: Option<anyhow::Error>,
}

/// One chat-only reviewer call with a single transient retry. Independent of
/// `Agent` so a panel can fan the same request out in parallel.
pub(crate) async fn run_review_attempt(
    provider: Arc<dyn Provider>,
    request: ChatRequest,
    timeout: Option<Duration>,
) -> ReviewAttempt {
    let mut attempts_left = 2u32;
    loop {
        attempts_left -= 1;
        let mut text = String::new();
        let mut sink = |event: StreamEvent| {
            if let StreamEvent::Text(chunk) = event {
                text.push_str(&chunk);
            }
        };
        match crate::agent::turn::await_side_call(
            timeout,
            provider.stream(request.clone(), &mut sink),
        )
        .await
        {
            Err(timeout) => {
                return ReviewAttempt {
                    verdict: SkepticVerdict::Unavailable(format!(
                        "provider timed out after {:.1}s",
                        timeout.as_secs_f64()
                    )),
                    usage: None,
                    error: None,
                };
            }
            Ok(Ok(completion)) => {
                if text.trim().is_empty() {
                    text = super::skeptic::content_text(&completion.content);
                }
                return ReviewAttempt {
                    verdict: super::skeptic::parse_verdict(&text),
                    usage: Some(completion.usage),
                    error: None,
                };
            }
            Ok(Err(err)) => {
                if attempts_left > 0 && super::skeptic::review_error_is_transient(&err) {
                    let delay = hi_ai::provider_retry_after_seconds(&err)
                        .unwrap_or(2)
                        .min(10);
                    tokio::time::sleep(Duration::from_secs(delay)).await;
                    continue;
                }
                return ReviewAttempt {
                    verdict: SkepticVerdict::Unavailable(format!("provider error: {err:#}")),
                    usage: None,
                    error: Some(err),
                };
            }
        }
    }
}

/// Production panel size is 3 (grok `GOAL_VERIFIER_SKEPTIC_COUNT`). Tests set 1.
pub(crate) const MAX_SKEPTIC_COUNT: u8 = 5;

pub(crate) fn clamp_skeptic_count(count: u8) -> u8 {
    count.clamp(1, MAX_SKEPTIC_COUNT)
}

/// Majority of parsed (non-unavailable) votes. Unavailable reviewers do not
/// count toward a refute.
pub(crate) fn aggregate_panel(verdicts: Vec<SkepticVerdict>) -> SkepticVerdict {
    let parsed: Vec<SkepticVerdict> = verdicts
        .into_iter()
        .filter(|verdict| !matches!(verdict, SkepticVerdict::Unavailable(_)))
        .collect();
    if parsed.is_empty() {
        return SkepticVerdict::Unavailable("panel produced no verdict".into());
    }
    let majority = parsed.len() / 2 + 1;
    let mut objections = Vec::new();
    let mut escalations = Vec::new();
    let mut object_n = 0usize;
    let mut escalate_n = 0usize;
    for verdict in &parsed {
        match verdict {
            SkepticVerdict::Object(items) => {
                object_n += 1;
                objections.extend(items.iter().cloned());
            }
            SkepticVerdict::Escalate(items) => {
                escalate_n += 1;
                escalations.extend(items.iter().cloned());
            }
            SkepticVerdict::Approve | SkepticVerdict::Unavailable(_) => {}
        }
    }
    if object_n >= majority {
        objections.dedup();
        return SkepticVerdict::Object(objections);
    }
    if escalate_n >= majority {
        escalations.dedup();
        return SkepticVerdict::Escalate(escalations);
    }
    SkepticVerdict::Approve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_objects_win_a_three_panel() {
        let verdict = aggregate_panel(vec![
            SkepticVerdict::Approve,
            SkepticVerdict::Object(vec!["missing CSRF path".into()]),
            SkepticVerdict::Object(vec!["stub in handler".into()]),
        ]);
        match verdict {
            SkepticVerdict::Object(items) => {
                assert!(items.iter().any(|item| item.contains("CSRF")));
                assert!(items.iter().any(|item| item.contains("stub")));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn one_object_cannot_refute() {
        assert_eq!(
            aggregate_panel(vec![
                SkepticVerdict::Approve,
                SkepticVerdict::Approve,
                SkepticVerdict::Object(vec!["nit".into()]),
            ]),
            SkepticVerdict::Approve
        );
    }

    #[test]
    fn tie_with_unavailable_fails_open() {
        assert_eq!(
            aggregate_panel(vec![
                SkepticVerdict::Approve,
                SkepticVerdict::Object(vec!["gap".into()]),
                SkepticVerdict::Unavailable("timeout".into()),
            ]),
            SkepticVerdict::Approve
        );
    }

    #[test]
    fn all_unavailable_stays_unavailable() {
        assert!(matches!(
            aggregate_panel(vec![
                SkepticVerdict::Unavailable("a".into()),
                SkepticVerdict::Unavailable("b".into()),
            ]),
            SkepticVerdict::Unavailable(_)
        ));
    }
}
