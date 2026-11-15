# H3 on M1–M4 before macOS 26.4

Compare a full 1024×1024 render on one Mac using the released portable MLX path and two experimental Metal kernels. The harness targets Apple Silicon M1 through M4, including Pro, Max and Ultra variants. A result applies to the CPU, GPU and macOS version recorded in that run.

## Run the comparison

Use macOS below 26.4 with working Metal, Xcode Command Line Tools, and an existing H3 INT8 model directory. Run from the root of the experimental checkout. The examples use uv to install the small runtime dependencies. The benchmark reads the model from the directory you supply.

```sh
git clone --branch experiment/legacy-mac-render https://github.com/Anionex/h3-image-mlx.git h3-legacy-bench
cd h3-legacy-bench
git worktree add --detach ../h3-simd 884608c72e3b489e3067ef1d862c264c395b2858
git worktree add --detach ../h3-row-energy 7b04f2432666a3ae248d6370901bc5ca0eebc951

uv run --no-project --python 3.13 --with 'mlx==0.32.4' --with 'h3-image-mlx==0.1.0a8' \
  python scripts/legacy_bench/compare.py \
  --candidate ../h3-simd --alternative ../h3-row-energy \
  --weights "/absolute/path/to/h3-models" --results ../h3-1024-results
```

Use a new results directory each time. For a six-render comparison of the SIMD candidate with the released runtime, omit `--alternative`. The default three-arm comparison runs nine images with a fixed prompt, seed 42 and the six-step H3 schedule. Each arm appears once in each position across three groups. Some Macs may take more than 30 minutes per image. Increase `--timeout` (seconds per render) when needed.

## Read the results

- `summary.json` gives the median end-to-end time in seconds and median peak stage-process footprint in bytes. For each render, the peak is the largest lifetime process footprint among text encoding, denoising and VAE decoding. Divide bytes by 1e9 for decimal GB.
- Every numbered directory contains the generated PNG and its SHA-256, plus uncached I/O counters and residency observations. All three stages must pass physical-read checks. The harness also checks that each PNG matches the released baseline on that machine.
- A warm weight cache causes the run to stop. On a personally controlled Mac, a reboot followed by running the benchmark before opening the model is one way to start cold. The harness samples file residency throughout generation.
- When sharing results, include `configuration.json`, `summary.json` and the numbered `result.json` files. They identify the hardware, macOS, runtime paths, file hashes, timing and measurement conditions. Runs using the optional `--proxy` flag are labeled as forced-dispatch screening.
