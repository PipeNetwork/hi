//! Mid-turn interjection and next-turn queue ownership.

use crate::App;

/// After `drive` finishes, align `app.queue` with what the agent consumed from
/// the interjection inbox.
///
/// Plain-text lines submitted mid-turn are pushed to both `app.queue` (visible,
/// next-turn FIFO) and the inbox (steer current turn). Anything the agent
/// drained must leave the queue so it does not run twice; leftovers stay queued.
pub(super) fn reconcile_queue_with_interjections(
    app: &mut App,
    inbox: &hi_agent::InterjectionInbox,
    commit_consumed: bool,
) {
    let leftover = inbox.drain();
    let offered: Vec<String> = app.mid_turn_offered.drain(..).collect();

    // A provider error or frontend cancellation can happen after the agent has
    // drained an interjection, but before the turn commits a usable result. In
    // that case the visible queue remains the source of truth: retain every
    // offered line so user work is retried by the next turn. Only successful
    // drive completion is allowed to remove inbox-consumed queue entries.
    if !commit_consumed {
        for msg in leftover {
            if !app.queue.iter().any(|q| q == &msg) {
                let _ = app.try_enqueue_prompt(msg);
            }
        }
        app.clamp_queue_selection();
        return;
    }

    if offered.is_empty() {
        // No dual-pushed lines; any stray inbox items still become next-turn work.
        for msg in leftover {
            let _ = app.try_enqueue_prompt(msg);
        }
        app.clamp_queue_selection();
        return;
    }
    // `leftover` must be a suffix of `offered` (both FIFO). Anything before that
    // suffix was applied mid-turn and should leave the visible queue.
    let consumed = if leftover.is_empty() {
        offered.len()
    } else if offered.len() >= leftover.len()
        && offered[offered.len() - leftover.len()..] == leftover[..]
    {
        offered.len() - leftover.len()
    } else {
        // Order diverged (user reordered/removed queue entries). Don't guess —
        // leave the queue as-is and append any true leftovers not already present.
        for msg in leftover {
            if !app.queue.iter().any(|q| q == &msg) {
                let _ = app.try_enqueue_prompt(msg);
            }
        }
        app.clamp_queue_selection();
        return;
    };
    for msg in offered.iter().take(consumed) {
        if app.queue.front() == Some(msg) {
            app.queue.pop_front();
            let _ = app.trace_prompt_dequeued(msg);
        } else if let Some(pos) = app.queue.iter().position(|q| q == msg) {
            // User may have reordered; still drop the consumed line once.
            let mut rest: std::collections::VecDeque<_> = app.queue.drain(pos..).collect();
            rest.pop_front();
            app.queue.append(&mut rest);
            let _ = app.trace_prompt_dequeued(msg);
        }
    }
    // `leftover` entries remain at the front of the queue from the original
    // dual-push (or were re-ordered); nothing more to enqueue.
    app.clamp_queue_selection();
}
