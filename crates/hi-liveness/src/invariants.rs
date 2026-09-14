//! Invariant helpers. These must never panic — unsupervised sessions share this path.

use crate::publisher::Publisher;
use crate::schema::InvariantCode;

pub fn report(publisher: &Publisher, code: InvariantCode) {
    publisher.set_invariant(code);
}
