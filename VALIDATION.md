# Local package validation

Earlier local prototype tested on 2026-11-11: `dist/h3_image_mlx-0.1.0a1-py3-none-any.whl`, 32,805 bytes. SHA-256: `08bdceebfc67c294c1551bdbca17caf42070bb92b10e13c3adb670a9872f5590`.

## Installed package

Built with `uv build --wheel --offline` and installed the resulting wheel into a fresh Python 3.13 environment using the local uv cache. The installed CLI exposes `generate`, `verify` and `fetch`. Tests run against the installed package checked all bundled data files, reference order and count validation, hardware dispatch, missing and corrupt files, and HTTP Range resumption against a mock server. The real model directory passed the four-file size check.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python dist/h3_image_mlx-0.1.0a1-py3-none-any.whl
.venv/bin/python -m unittest discover -s tests -p test_distribution.py -v
H3_IMAGE_TEST_WEIGHTS=/path/to/models .venv/bin/python -m unittest discover -s tests -p test_portable.py -v
```

## Compute verification

On an M5 Pro with 48 GB of unified memory, two sampled INT8 projections from the actual DiT and text encoder were tested on the portable MLX path. All 16,640 BF16 outputs matched both the native W8A8 path and the CPU implementation using integer dot products. Tests also cover operation when creation of a custom Metal kernel is disabled.

Hardware detection selects the portable path for M1, M2, M3 and M4. The tests described here ran on an M5 Pro. Physical validation on each earlier chip is outstanding.

## End-to-end inference with the installed wheel

A second isolated Python 3.13 environment used MLX 0.32.3, the local weights, and the same wheel. With seed 42, the native and portable paths both generated a 512×512 tea-set photograph using the streamed DiT. CLI wall time included interpreter startup and writing the PNG.

| Backend requested | Hardware | Time | RGB SHA-256 |
| --- | --- | ---: | --- |
| auto (W8A8) | M5 Pro | 28.48 s | `a6ae77d0560852f21e8d86e2797bad25da599d25f00f9b0dd093d72ac9c0768a` |
| portable | M5 Pro | 56.94 s | `a6ae77d0560852f21e8d86e2797bad25da599d25f00f9b0dd093d72ac9c0768a` |

All RGB channels matched at all 262,144 pixels on this prompt and seed. Each timing represents one run.

The installed Python API also completed a 1024×1024 edit in 92.61 seconds on the M5 Pro. The first reference supplied the teapot scene and the second supplied the ceramic bird. All output RGB values matched the previously accepted desktop image for the same inputs and seed (SHA-256 of decoded RGB: `1499f0f465353e2c4eef22b8db67cae26a2e3fbd256b0bc759b1a4eec663a2fd`).

## Public 0.1.0a1 release

The release artifacts contain the runtime and download manifest. Both omit weights and VAE calibration data. The 1,826-byte upstream VAE configuration matches on both download sources and supplies the same 48 normalization values used in the prototype. The installed release wheel passed all nine tests with local weights and completed a 512×512 single-reference edit using the external VAE configuration on M5 Pro. Both artifacts passed strict Twine validation. See [the release hashes](releases/v0.1.0a1.sha256).

## Public 0.1.0a2 release

Validated on 2026-11-12 with Python 3.13.13, MLX 0.32.4 and the existing model files on M5 Pro (48 GB). All nine tests passed against the installed wheel, including the real DiT and text-encoder projection checks. The wheel and source distribution passed strict Twine validation and remain runtime-only. See [the release hashes](releases/v0.1.0a2.sha256).

The installed Python API completed four serial 512×512 checks using W8A8, streaming and seed 42:

- Default generation with all performance switches unset.
- Generation with `H3_ASYNC=1 H3_PREFETCH=1`, profiling and an initially empty embedding cache.
- The same request with a cache hit: only the DiT and VAE stages ran.
- A single-reference edit with both streaming switches enabled.

The first three outputs were pixel-identical (decoded RGB SHA-256: `e36333e6319536ba10bcdd459fd91e0f9b3d6a9485bac911cc48610e2d72e906`). Profiling recorded six denoising steps and nonzero weight reads. The reference edit completed text, reference encoding, DiT and VAE stages and returned a 512×512 RGB image. These were functional release checks; the performance comparisons are documented separately in the [performance guide](docs/PERFORMANCE.md#reported-results).
