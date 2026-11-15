// One SIMD group sums up to 1024 int8 magnitudes in one activation row.
const uint group = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
const uint width = x_shape[1], start = group * 1024;
uint sum = 0;
for (uint k = start + thread_index_in_simdgroup; k < min(width, start + 1024u); k += 32) {
  sum += abs(int(x[size_t(row) * width + k]));
}
sum = simd_sum(sum);
if (thread_index_in_simdgroup == 0) bounds[row * ((width + 1023) / 1024) + group] = sum;
