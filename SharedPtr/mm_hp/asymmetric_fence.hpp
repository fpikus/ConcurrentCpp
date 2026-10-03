// Copyright (c) 2026 Maged Michael
// See LICENSES for licensing terms.

#pragma once

#include <atomic>
#include <cstdlib>
#include <linux/membarrier.h>
#include <sys/syscall.h>
#include <unistd.h>

/// Asymmetric Memory Fences

namespace p1202 {

inline void asymmetric_thread_fence_light() noexcept {
  std::atomic_signal_fence(std::memory_order::seq_cst);
}

inline void asymmetric_thread_fence_heavy() noexcept {
  static const bool supported = ::syscall(SYS_membarrier,
      MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0) == 0;
  // This implementation assumes membarrier support.
  // For portable implementations, see folly AsymmetricThreadFence.
  if (!supported) std::abort();
  ::syscall(SYS_membarrier, MEMBARRIER_CMD_PRIVATE_EXPEDITED, 0, 0);
}

} // namespace p1202
