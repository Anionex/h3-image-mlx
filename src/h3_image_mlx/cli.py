import argparse
import os
from pathlib import Path
import sys
import tempfile

from .models import check_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="MiniMax H3 Image on Apple Silicon")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate an image or edit ordered references")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output", type=Path, required=True, help="Destination PNG")
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--reference", type=Path, action="append", default=[], help="Repeat in <Picture n> order")
    gen.add_argument("--backend", choices=("auto","w8a8","portable"), default="auto")
    gen.add_argument("--dit-mode", choices=("auto","stream","resident"), default="auto")
    get = commands.add_parser("fetch", help="Download the pinned weights, resuming partial files")
    get.add_argument("--source", choices=("modelscope","huggingface"), default="modelscope")
    verify = commands.add_parser("verify", help="Verify all five local file hashes")
    for command in (gen, get, verify):
        command.add_argument("--models", type=Path, required=True, help="Directory containing the pinned model files")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            from .download import fetch
            print(fetch(args.models, args.source))
        elif args.command == "verify":
            print(check_models(args.models, hashes=True))
        else:
            from .pipeline import generate
            if args.output.suffix.lower() != ".png":
                raise ValueError("Output must be a .png file")
            image = generate(args.prompt, models=args.models, width=args.width, height=args.height,
                             seed=args.seed, references=args.reference, backend=args.backend, dit_mode=args.dit_mode)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
                try:
                    image.save(tmp, format="PNG")
                    tmp.flush()
                    os.replace(tmp.name, args.output)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            print(args.output)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
