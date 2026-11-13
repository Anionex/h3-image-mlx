# Model and runtime boundaries

The Python package owns public requests, model selection, pinned downloads and results. A model adapter owns that model's validation, conditioning syntax, sampling policy and execution plan. Runtime workers own tensor execution. Desktop UI, HTTP authentication, histories and application process state are outside this library.

```
CLI / Python API
  → validated model request
  → model registry
      H3 adapter → isolated Python/MLX stages
      Qwen adapter → isolated native MLX stages or full request
  → image + structured metrics + progress/cancellation
```

The existing `generate(prompt, models=...)` API continues to select H3. Explicit Qwen selection uses `model="qwen-image-2.1-turbo"` and typed `QwenOptions`. Qwen-only arguments are rejected on H3 rather than ignored. Model weights and adapter assets use the same resume/hash/disk-space download machinery, with separate manifests and licenses.

Qwen's native runtime is an optional platform package, built from a pinned upstream revision and the checked-in source patch. It is independent of the desktop app and has no HTTP server. The main Python package remains installable without the native package for H3 users. Binary payloads, libraries, Metal resources and licenses are bundled together; they never depend on Homebrew or another application's installation at runtime.

A Qwen execution plan chooses the three-process path for ordinary Turbo eight-step generation. Editing, variation, CFG, custom sigma schedules and LoRA use the full-request entry point, which preserves the same model implementations and loader policies. The public API exposes progress and cancellation in both cases. A full request has a different memory lifetime and is measured separately.

LoRA presets resolve to an explicit transformer view, sigma schedule and ordered adapter list. Additional adapters retain independent scales. Prepared DiT blocks are keyed by the actual transformer source, not merely a model name; changing a transformer must not reuse another model's prepared weights. Private model views never overwrite source weights. Derived assets and preparation costs are recorded separately from generation.

Benchmark cache policy is independent of runtime tuning. Low-memory timing claims require verified uncached weight reads. Cached measurements remain separately labelled. Disk-backed prepared weights are persistent derived files; they are not permission to keep unmeasured model-weight pages in RAM.
