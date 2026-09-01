//! Conservative announcement startup and explicit persistence commands.

use anyhow::{Result, bail};
use hi_announcements::{
    AnnouncementEndpointConfig, AnnouncementSeverity, RemoteAnnouncement, announcement_hide_key,
    fetch_announcements, filter_expired, mutate_hidden_announcement_ids,
    read_hidden_announcement_ids, resolve_startup_announcements, visible_announcements,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn home() -> PathBuf {
    crate::session::data_root().unwrap_or_else(|| PathBuf::from(".hi"))
}

async fn configured() -> Option<Vec<RemoteAnnouncement>> {
    // The override never touches the network, preserving deterministic tests and
    // managed-launcher behavior.
    if std::env::var_os("HI_ANNOUNCEMENTS_OVERRIDE").is_some() {
        return resolve_startup_announcements(Ok(Vec::new())).map(filter_expired);
    }
    let endpoint = std::env::var("HI_ANNOUNCEMENTS_ENDPOINT").ok()?;
    let fetched = fetch_announcements(home(), &AnnouncementEndpointConfig::https(endpoint)).await;
    resolve_startup_announcements(fetched.map(|refresh| refresh.announcements)).map(filter_expired)
}

async fn load() -> (Vec<RemoteAnnouncement>, BTreeSet<String>) {
    let announcements = configured().await.unwrap_or_default();
    let hidden = read_hidden_announcement_ids(home())
        .await
        .unwrap_or_default();
    (announcements, hidden)
}

fn render(announcements: &[RemoteAnnouncement], hidden: &BTreeSet<String>) -> Vec<String> {
    visible_announcements(&filter_expired(announcements.iter().cloned()))
        .into_iter()
        .filter(|announcement| !hidden.contains(&announcement_hide_key(announcement)))
        .map(|announcement| {
            let severity = match announcement.severity {
                AnnouncementSeverity::Info => "info",
                AnnouncementSeverity::Warning => "warning",
                AnnouncementSeverity::Critical => "critical",
            };
            let id = announcement_hide_key(announcement);
            let title = announcement
                .title
                .as_deref()
                .map(|title| format!("{title}: "))
                .unwrap_or_default();
            format!(
                "[{severity}] {title}{} ({id})",
                announcement.message.as_deref().unwrap_or_default().trim()
            )
        })
        .collect()
}

pub(crate) type PendingAnnouncements =
    tokio::task::JoinHandle<(Vec<RemoteAnnouncement>, BTreeSet<String>)>;

/// Start the fetch immediately so network and disk latency overlap the session
/// instead of delaying startup. Display — and the auto-hide that must only
/// follow an actual display — happens later via `show_detached` or
/// `show_after_session`.
pub(crate) fn spawn_load() -> PendingAnnouncements {
    tokio::spawn(load())
}

/// Print whenever the load completes: for plain-terminal sessions (REPL),
/// where output stays visible no matter when it interleaves.
pub(crate) fn show_detached(pending: PendingAnnouncements) {
    tokio::spawn(async move {
        if let Ok(loaded) = pending.await {
            show(loaded).await;
        }
    });
}

/// Print after a full-screen session has returned to the main screen. Printing
/// before or during the TUI would land in the alternate screen, be erased on
/// the first redraw, and still mark one-shot announcements as seen. The fetch
/// has had the whole session to finish; the short wait matters only when the
/// user quits immediately.
pub(crate) async fn show_after_session(pending: PendingAnnouncements) {
    if let Ok(Ok(loaded)) = tokio::time::timeout(std::time::Duration::from_secs(1), pending).await {
        show(loaded).await;
    }
}

async fn show((announcements, hidden): (Vec<RemoteAnnouncement>, BTreeSet<String>)) {
    let lines = render(&announcements, &hidden);
    for line in &lines {
        eprintln!("\x1b[33mannouncement: {line}\x1b[0m");
    }
    if !lines.is_empty() {
        eprintln!("\x1b[2mmanage with `hi announcements list|dismiss <id>`\x1b[0m");
    }
    // Non-persistent entries are one-display notices. Persistent entries stay
    // visible until explicit dismissal or expiry.
    let displayed_non_persistent = visible_announcements(&filter_expired(announcements))
        .into_iter()
        .filter(|announcement| {
            !announcement.persistent && !hidden.contains(&announcement_hide_key(announcement))
        })
        .map(announcement_hide_key)
        .collect::<BTreeSet<_>>();
    if !displayed_non_persistent.is_empty() {
        let home = home();
        let _ = mutate_hidden_announcement_ids(home, |hidden| {
            hidden.extend(displayed_non_persistent);
        })
        .await;
    }
}

pub(crate) async fn run_cli(args: &[String]) -> Result<()> {
    let (announcements, hidden) = load().await;
    match args.first().map(String::as_str).unwrap_or("list") {
        "list" => {
            let lines = render(&announcements, &hidden);
            if lines.is_empty() {
                println!("No visible announcements.");
            } else {
                for line in lines {
                    println!("{line}");
                }
            }
        }
        "dismiss" => {
            let Some(id) = args.get(1).filter(|id| !id.trim().is_empty()) else {
                bail!("usage: hi announcements dismiss <id>");
            };
            let Some(announcement) = filter_expired(announcements.iter().cloned())
                .into_iter()
                .find(|announcement| announcement_hide_key(announcement) == *id)
            else {
                bail!("no active announcement with id '{id}'");
            };
            if !announcement.dismissible {
                bail!("announcement '{id}' cannot be dismissed");
            }
            mutate_hidden_announcement_ids(home(), |hidden| hidden.insert(id.clone())).await?;
            println!("Dismissed announcement {id}.");
        }
        other => bail!("unknown announcements command '{other}'; use list or dismiss <id>"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_hides_dismissed_and_empty_entries() {
        let announcements = vec![
            RemoteAnnouncement {
                id: Some("visible".into()),
                message: Some("Hello".into()),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("hidden".into()),
                message: Some("Secret".into()),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("empty".into()),
                message: Some("  ".into()),
                ..Default::default()
            },
        ];
        let lines = render(&announcements, &BTreeSet::from(["hidden".into()]));
        assert_eq!(lines, ["[info] Hello (visible)"]);
    }
}
