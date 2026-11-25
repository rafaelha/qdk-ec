//! What decodes wait for and read, so a caller can tell a missing graph edge from slow work.

use super::WindowCoordinator;
use hashbrown::HashSet;
use tokio::sync::watch;

/// The waits of one decode, keyed by its gid.
///
/// `ports` and `error_models` are cleared when the decode returns; `dependencies` stay until
/// reset, so a caller can still ask what a finished decode read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodeWait {
    /// Unconnected `(gid, output_port)` pairs the mandatory zone is waiting on.
    pub ports: Vec<(u64, u64)>,
    /// Gadgets whose error models the decode is waiting on.
    pub error_models: Vec<u64>,
    /// The decode's mandatory zone, sorted: the gadgets it waits for, not the best-effort lookahead.
    pub dependencies: Vec<u64>,
}

/// Clears a decode's waits when the decode returns, however it returns.
pub(super) struct DecodeWaitGuard<'a> {
    coordinator: &'a WindowCoordinator,
    gid: u64,
}

impl Drop for DecodeWaitGuard<'_> {
    fn drop(&mut self) {
        if let Some(wait) = self.coordinator.decode_waits.lock().unwrap().get_mut(&self.gid) {
            wait.ports.clear();
            wait.error_models.clear();
        }
        self.coordinator.decode_revision.send_modify(|revision| *revision += 1);
    }
}

impl WindowCoordinator {
    /// Track `gid`'s waits until the guard drops, and its dependencies until reset.
    pub(super) fn track_decode(&self, gid: u64) -> DecodeWaitGuard<'_> {
        self.decode_waits.lock().unwrap().insert(
            gid,
            DecodeWait {
                dependencies: vec![gid],
                ..Default::default()
            },
        );
        self.decode_revision.send_modify(|revision| *revision += 1);
        DecodeWaitGuard { coordinator: self, gid }
    }

    /// Update `gid`'s waits if its decode is tracked.
    pub(super) fn update_decode_wait(&self, gid: u64, update: impl FnOnce(&mut DecodeWait)) {
        let changed = self.decode_waits.lock().unwrap().get_mut(&gid).is_some_and(|wait| {
            let before = wait.clone();
            update(wait);
            *wait != before
        });
        if changed {
            self.decode_revision.send_modify(|revision| *revision += 1);
        }
    }

    pub(super) fn sorted(gids: &HashSet<u64>) -> Vec<u64> {
        let mut gids: Vec<u64> = gids.iter().copied().collect();
        gids.sort_unstable();
        gids
    }

    /// The gadgets in `gid`'s mandatory zone, or `[gid]` before its decode explores one.
    pub async fn decode_dependencies(&self, gid: u64) -> Vec<u64> {
        self.decode_waits
            .lock()
            .unwrap()
            .get(&gid)
            .map_or_else(|| vec![gid], |wait| wait.dependencies.clone())
    }

    /// `gid`'s waits, with ports connected since dropped, or `None` once its readouts are final.
    pub(crate) async fn decode_wait(&self, gid: u64) -> Option<DecodeWait> {
        let mut wait = self.decode_waits.lock().unwrap().get(&gid).cloned().unwrap_or_default();
        {
            let gadgets = self.gadgets.read().await;
            match gadgets.get(&gid) {
                Some(gadget) if gadget.pauli_frame.borrow().is_some() => return None,
                Some(_) => {}
                None if self.collected_gids.lock().await.contains(&gid) => return None,
                None => {}
            }
            wait.ports.retain(|&(source, port)| {
                gadgets
                    .get(&source)
                    .is_some_and(|gadget| gadget.outputs[usize::try_from(port).unwrap()].borrow().is_none())
            });
        }
        Some(wait)
    }

    /// Changes whenever a decode's waits or readiness may have changed.
    pub(crate) fn subscribe_decode_progress(&self) -> watch::Receiver<u64> {
        self.decode_revision.subscribe()
    }
}
