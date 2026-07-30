//! Always-on per-window decode timing log. Mirrors `dem::DemLog`'s shape
//! (short std::sync::Mutex critical sections, no .await inside) but has no
//! enable flag — records are ~30 scalars per window, negligible next to a
//! decode call. Drained via the DrainWindowTimings RPC; reset() clears it so
//! the buffer stays bounded to one shot even with no consumer attached.

use std::sync::Mutex;

use crate::coordinator::WindowTiming;

#[derive(Default)]
struct Inner {
    next_seq: u64,
    records: Vec<WindowTiming>,
}

#[derive(Default)]
pub struct TimingLog {
    inner: Mutex<Inner>,
}

impl TimingLog {
    /// Push a completed record; `seq` is assigned here in push order.
    pub fn push(&self, mut timing: WindowTiming) {
        let mut inner = self.inner.lock().unwrap();
        timing.seq = inner.next_seq;
        inner.next_seq += 1;
        inner.records.push(timing);
    }

    /// Take everything recorded since the previous drain.
    pub fn drain(&self) -> Vec<WindowTiming> {
        std::mem::take(&mut self.inner.lock().unwrap().records)
    }

    /// Forget everything and restart seq (coordinator reset between shots).
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        *inner = Inner::default();
    }
}
