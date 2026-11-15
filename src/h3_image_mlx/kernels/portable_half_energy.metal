// Each threadgroup sums one half of one int8 row. CUT is on a 64-value boundary.
const uint half_id = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y, width = x_shape[1];
const uint begin = half_id ? CUT : 0, end = half_id ? width : CUT;
uint sum = 0;
for (uint k = begin + thread_position_in_threadgroup.x; k < end; k += 128) {
    const int v = x[size_t(row) * width + k];
    sum += uint(v * v);
}
threadgroup uint groups[4];
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0) groups[simdgroup_index_in_threadgroup] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (thread_position_in_threadgroup.x == 0)
    energies[row * 2 + half_id] = groups[0] + groups[1] + groups[2] + groups[3];
