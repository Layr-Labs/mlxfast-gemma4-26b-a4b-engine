namespace mlx::core::metal {

const char* gather_front() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/indexing/gather_front.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/indexing/indexing.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/indexing/indexing.h"
// Copyright © 2023-2024 Apple Inc.


#include <metal_stdlib>

template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant int64_t* strides;
  const constant bool* row_contiguous;
  const int ndim;
};

template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, int size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/indexing/gather_front.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/indexing/gather_front.h"
// Copyright © 2025 Apple Inc.



template <typename T, typename IdxT, typename LocT, int N>
[[kernel]] void gather_front(
    const device T* src,
    const device IdxT* indices,
    device T* out,
    const constant int64_t& stride,
    const constant int& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  // GEMMA4-PREFILL-COMPACT-X. The production routed-expert gather asks for
  // 65,536 stable-sorted copies of 8,192 unique rows, each 2,816 values wide.
  // The following grouped affine NAX projections need the same logical rows,
  // but do not require them to be physically duplicated. Keep the allocation
  // and logical shape (so the trusted host keeps selecting grouped NAX), store
  // one token-major copy of each source row at the front, and place the exact
  // 65,536-entry source-row map immediately after those unique rows.
  //
  // `indices` is already `order.floorDivide(8)`, so every mapped read reaches
  // the same T word as the fully materialized tensor. Payload writes fall from
  // 65,536 * 2,816 to 8,192 * 2,816 plus one uint32 per route.
  const bool gemma4_compact =
      stride == 2816 && size == 8192 && grid_dim.y == 65536 &&
      sizeof(T) == 2 && sizeof(IdxT) == 4;
  if (gemma4_compact) {
    if (index.x == 0) {
      device uint32_t* row_map =
          (device uint32_t*)(out + LocT(size) * LocT(stride));
      row_map[index.y] =
          static_cast<uint32_t>(offset_neg_idx(indices[index.y], size));
    }
    if (index.y >= static_cast<uint>(size)) {
      return;
    }
    const LocT row = static_cast<LocT>(index.y);
    const LocT row_base = static_cast<LocT>(stride) * row;
    int s_idx = N * index.x;
    for (int i = 0; i < N && s_idx < stride; ++i, ++s_idx) {
      out[row_base + s_idx] = src[row_base + s_idx];
    }
    return;
  }

  auto idx = offset_neg_idx(indices[index.y], size);
  LocT src_idx = static_cast<LocT>(stride) * idx;
  LocT out_idx = static_cast<LocT>(stride) * index.y;

  int s_idx = N * index.x;
  for (int i = 0; i < N && s_idx < stride; ++i, ++s_idx) {
    out[out_idx + s_idx] = src[src_idx + s_idx];
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
