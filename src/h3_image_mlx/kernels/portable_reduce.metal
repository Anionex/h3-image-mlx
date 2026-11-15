// Each plane is the exact dot product of at most 1024 signed int8 pairs.
const uint idx = thread_position_in_grid.x;
const uint cols = dots_shape[2], plane_size = dots_shape[1] * cols;
if (idx >= plane_size) return;
int total = 0;
for (int plane = 0; plane < dots_shape[0]; ++plane) {
  total += int(dots[size_t(plane) * plane_size + idx]);
}
out[idx] = T(float(total) * xs[idx / cols] * ws[idx % cols]);
