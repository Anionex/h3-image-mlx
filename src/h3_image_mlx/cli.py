"""CLI model selection shares the same typed requests as the Python API."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .api import (
    AUDIO_MODEL_IDS, MODEL_IDS, PINNED_MODEL_IDS,
    Lora, QwenOptions, generate_audio_result, generate_result,
)
from .models import check_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="Image and audio generation on Apple Silicon")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate or edit ordered references")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--reference", type=Path, action="append", default=[])
    gen.add_argument("--backend", choices=("auto", "w8a8", "portable"), default="auto")
    gen.add_argument(
        "--dit-mode", choices=("auto", "stream", "resident"), default="auto"
    )
    gen.add_argument("--metrics", type=Path)
    gen.add_argument("--progress", action="store_true")
    gen.add_argument("--weight-cache", choices=("system", "uncached"))
    gen.add_argument(
        "--stream-execution", choices=("sync", "overlap"),
        help="H3 and Z-Image block scheduling. For Qwen, use --qwen-dit-execution.",
    )
    gen.add_argument("--steps", type=int)
    q = gen.add_argument_group("Qwen Image 2.1 options")
    q.add_argument(
        "--memory-mode", choices=("resident", "prefetch", "micro-fast", "micro")
    )
    q.add_argument("--runtime", type=Path)
    q.add_argument("--cache", type=Path)
    q.add_argument("--prepared", type=Path)
    q.add_argument(
        "--no-prepare",
        action="store_true",
        help="Read original weights without an extra prepared DiT copy",
    )
    q.add_argument("--qwen-dit-execution", choices=("native", "stream-sync", "stream-overlap"))
    q.add_argument("--cfg", type=float)
    q.add_argument("--negative-prompt")
    q.add_argument(
        "--profile",
        choices=(
            "complete",
            "turbo4",
            "turbo3",
            "standard",
            "lightning",
            "dmd2",
            "turbo8",
        ),
    )
    q.add_argument("--profile-assets", type=Path)
    q.add_argument("--profile-strength", type=float)
    q.add_argument("--auto-strength", action="store_true")
    q.add_argument("--lora", type=Path, action="append", default=[])
    q.add_argument("--lora-scale", type=float, action="append", default=[])
    q.add_argument("--sigmas", type=float, nargs="+")
    q.add_argument("--reference-mode", choices=("edit", "variation"))
    q.add_argument("--strength", type=float)
    q.add_argument("--reference-resolution", type=int)
    q.add_argument("--transparent", action="store_true")
    q.add_argument(
        "--private",
        action="store_true",
        help="Pass request data through pipes, without temporary prompt/image files",
    )
    get = commands.add_parser("fetch", help="Download pinned model weights")
    get.add_argument(
        "--source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    verify = commands.add_parser("verify", help="Verify pinned file hashes")
    align = commands.add_parser(
        "align", help="Replace the DiT and text encoder by page-aligned copies for zero-copy loading (H3_ZERO_COPY=1)"
    )
    align.add_argument("--keep-original", action="store_true")
    for command in (gen, get, verify, align):
        command.add_argument("--models", type=Path, required=True)
        command.add_argument("--model", choices=MODEL_IDS if command is gen else PINNED_MODEL_IDS, default="h3")
        command.add_argument("--precision", choices=("int8", "bf16"), default="int8")
    audio = commands.add_parser("audio", help="Generate speech or sound effects")
    audio.add_argument("--model", choices=AUDIO_MODEL_IDS, required=True)
    audio.add_argument("--models", type=Path, required=True)
    audio.add_argument("--prompt", required=True)
    audio.add_argument("--output", type=Path, required=True)
    audio.add_argument("--seed", type=int, default=42)
    audio.add_argument("--seconds", type=float, help="Sound-effect duration in seconds")
    audio.add_argument("--steps", type=int, help="Sound-effect denoising steps")
    audio.add_argument("--language", help="Ming speech language code, for example zh")
    audio.add_argument("--voice", help="Built-in Ming voice, for example US_James or 灵小柔")
    audio.add_argument("--weight-cache", choices=("system", "uncached"), default="system")
    audio.add_argument("--stream-execution", choices=("sync", "overlap"))
    audio.add_argument("--metrics", type=Path)
    presets = commands.add_parser(
        "fetch-profile", help="Download a Qwen sampling preset and base DiT"
    )
    presets.add_argument(
        "--profile", required=True, choices=("standard", "lightning", "dmd2", "turbo8")
    )
    presets.add_argument("--directory", type=Path, required=True)
    presets.add_argument(
        "--source", choices=("modelscope", "huggingface"), default="modelscope"
    )
    args = parser.parse_args(argv)
    try:
        if args.command in ("generate", "audio") and args.metrics is not None:
            if args.output.resolve() == args.metrics.resolve():
                raise ValueError("--output and --metrics must point to different files")
        if args.command == "audio":
            return _audio(args)
        if args.command == "fetch-profile":
            from .qwen.assets import fetch_profile

            print(fetch_profile(args.directory, args.profile, args.source))
            return 0
        if args.model != "h3" and args.precision != "int8":
            raise ValueError("Select a checkpoint for this model to change its precision")
        if args.command == "fetch":
            if args.model == "h3":
                from .download import fetch

                print(fetch(args.models, args.source, precision=args.precision))
            else:
                from .qwen.assets import fetch

                print(fetch(args.models, args.source))
        elif args.command == "align":
            from .align import align_models

            print(align_models(args.models, args.precision, args.keep_original))
        elif args.command == "verify":
            if args.model == "h3":
                print(check_models(args.models, hashes=True, precision=args.precision))
            else:
                from .qwen.assets import check

                print(check(args.models, hashes=True))
        else:
            if args.output.suffix.lower() != ".png":
                raise ValueError("Output must be a .png file")
            fields = {
                "memory_mode": args.memory_mode,
                "dit_execution": args.qwen_dit_execution,
                "runtime": args.runtime,
                "cache": args.cache,
                "prepared": args.prepared,
                "steps": args.steps if args.model != "z-image-turbo" else None,
                "guidance_scale": args.cfg,
                "negative_prompt": args.negative_prompt,
                "profile": args.profile,
                "profile_assets": args.profile_assets,
                "profile_strength": args.profile_strength,
                "sample_sigmas": args.sigmas,
                "mode": args.reference_mode,
                "strength": args.strength,
                "reference_resolution": args.reference_resolution,
            }
            fields = {k: v for k, v in fields.items() if v is not None}
            if args.no_prepare:
                fields["prepare_weights"] = False
            for name in ("auto_strength", "transparent", "private"):
                if getattr(args, name):
                    fields[name] = True
            if args.lora_scale and len(args.lora_scale) != len(args.lora):
                raise ValueError(
                    "Provide one --lora-scale per --lora, or omit all scales for 1.0"
                )
            if args.lora:
                fields["loras"] = tuple(
                    Lora(p, s)
                    for p, s in zip(
                        args.lora, args.lora_scale or [1.0] * len(args.lora)
                    )
                )
            is_qwen = args.model == "qwen-image-2.1-turbo"
            if not is_qwen and fields:
                raise ValueError("Qwen options require --model qwen-image-2.1-turbo")
            progress = (
                (lambda e: print(json.dumps(e), file=sys.stderr, flush=True))
                if args.progress
                else None
            )
            result = generate_result(
                args.prompt,
                models=args.models,
                model=args.model,
                width=args.width,
                height=args.height,
                seed=args.seed,
                references=args.reference,
                backend=args.backend,
                dit_mode=args.dit_mode,
                precision=args.precision,
                weight_cache=args.weight_cache,
                steps=args.steps if args.model == "z-image-turbo" else None,
                overlap_streaming=(
                    args.stream_execution == "overlap" if args.stream_execution is not None else None
                ),
                qwen=QwenOptions(**fields) if is_qwen else None,
                on_progress=progress,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=args.output.parent, delete=False
            ) as tmp:
                try:
                    result.image.save(tmp, format="PNG")
                    tmp.flush()
                    os.replace(tmp.name, args.output)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            if args.metrics:
                args.metrics.parent.mkdir(parents=True, exist_ok=True)
                args.metrics.write_text(json.dumps(result.metrics, indent=2))
            print(args.output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


def _audio(args):
    if args.output.suffix.lower() != ".wav":
        raise ValueError("Audio output must be a .wav file")
    options = {name: getattr(args, name) for name in ("seconds", "steps") if getattr(args, name) is not None}
    if args.model != "moss-soundeffect-v2" and options:
        raise ValueError("--seconds and --steps apply to MOSS SoundEffect")
    if args.language is not None or args.voice is not None:
        if args.model != "ming-3.0-tts":
            raise ValueError("--language and --voice apply to Ming speech")
        if args.language is not None:
            options["lang_code"] = args.language
        if args.voice is not None:
            options["voice"] = args.voice

    import mlx.core as mx
    import numpy as np
    import soundfile

    result = generate_audio_result(
        args.prompt, models=args.models, model=args.model, seed=args.seed,
        weight_cache=args.weight_cache,
        overlap_streaming=args.stream_execution == "overlap" if args.stream_execution is not None else None,
        **options,
    )
    samples = np.asarray(result.waveform.astype(mx.float32)).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
        try:
            soundfile.write(tmp.name, samples, result.sample_rate, format="WAV", subtype="FLOAT")
            os.replace(tmp.name, args.output)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(result.metrics, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
