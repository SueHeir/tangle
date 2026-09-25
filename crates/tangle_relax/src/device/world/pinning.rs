//! Per-vertex kinematic pins set between batches.

use cubecl::prelude::*;

use super::DeviceFiberWorld;

impl<R: Runtime> DeviceFiberWorld<R> {
    /// Replaces the per-vertex pin flags (one per packed vertex; nonzero pins).
    ///
    /// A pinned vertex keeps its position through contact, stretch, bend,
    /// curvature and image-force corrections, but its segments still take
    /// part in contact, so pinned fibers act as fixed obstacles for the rest.
    ///
    /// # Panics
    ///
    /// Panics unless `pinned` has one entry per packed vertex.
    pub fn set_vertex_pinned(&mut self, pinned: &[u32]) {
        assert_eq!(
            pinned.len(),
            self.packed.vertex_count(),
            "one pin flag per packed vertex"
        );
        let flags: Vec<u32> = pinned.iter().map(|&flag| u32::from(flag != 0)).collect();
        self.vertex_pinned = self.client.create_from_slice(u32::as_bytes(&flags));
        self.packed.vertex_pinned = flags;
    }

    /// Number of currently pinned packed vertices.
    pub fn pinned_vertex_count(&self) -> usize {
        self.packed
            .vertex_pinned
            .iter()
            .filter(|&&flag| flag != 0)
            .count()
    }
}
