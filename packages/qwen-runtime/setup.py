import json
from pathlib import Path
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        manifest = (
            Path(__file__).parent / "src/h3_image_mlx_native/payload/runtime.json"
        )
        if not manifest.is_file():
            raise RuntimeError(
                "Run scripts/build_qwen_runtime.py before building the native wheel"
            )
        platform = json.loads(manifest.read_text()).get(
            "wheel_platform", "macosx_26_0_arm64"
        )
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": NativeWheel})
