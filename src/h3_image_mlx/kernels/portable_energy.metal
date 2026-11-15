// One threadgroup per row. For a supported width <= 25600 the sum fits in uint32.
const uint row = threadgroup_position_in_grid.x, width = x_shape[1];
uint sum = 0;
for (uint i = thread_position_in_threadgroup.x; i < width; i += 128) {
  const int v = x[size_t(row) * width + i];
  sum += uint(v * v);
}
sum = simd_sum(sum);
threadgroup uint partial[4];
if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (thread_position_in_threadgroup.x == 0)
  squares[row] = partial[0] + partial[1] + partial[2] + partial[3];
