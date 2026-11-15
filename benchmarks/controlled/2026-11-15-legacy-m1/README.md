# H3 large Metal projections on a virtual Apple M1

Each CI job rendered the same 1024×1024 prompt six times on a single macOS 15.8.2 runner with an Apple Paravirtual GPU and MLX 0.32.4. The baseline was the portable backend from PyPI release `0.1.0a8`. The candidates ran in separate jobs, each with its own baseline. Generation used six steps, seed 42 and streamed INT8 weights.

| Experiment | Baseline | Candidate | Time reduction | Baseline peak | Candidate peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 bound only | 788.72 s | 751.64 s | 4.7% | 1.904 GB | 1.904 GB |
| 64 × 64 SIMD | 661.78 s | 324.26 s | 51.0% | 1.904 GB | 1.752 GB |

Times and peaks are the medians of three runs per implementation. Each peak is the largest lifetime footprint across the text encoder, DiT and VAE stage processes in decimal GB. The run order was baseline, candidate, candidate, baseline, baseline, candidate. All six PNGs in each job matched byte for byte. File residency and guest block-device reads were checked throughout each render. Host-side disk caching was outside the guest measurements.

## Measurements

- L1 bound only: [Per-render timings, stage peaks, disk reads and PNG hashes](l1.json) · [CI job](https://github.com/Anionex/h3-image-mlx/actions/runs/43678325106)
- 64 × 64 SIMD: [Per-render timings, stage peaks, disk reads and PNG hashes](simd.json) · [CI job](https://github.com/Anionex/h3-image-mlx/actions/runs/43681597496)
