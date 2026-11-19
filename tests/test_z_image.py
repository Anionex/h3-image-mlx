import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from h3_image_mlx.adapters.z_image import generate_z_image


@unittest.skipUnless(find_spec("mflux"), "requires the optional MFlux runtime")
class ZImageCheckpointTests(unittest.TestCase):
    def test_accepts_mflux_quantization_metadata(self):
        from mflux.models.z_image.variants.z_image import ZImage
        from mflux.models.z_image.z_image_initializer import ZImageInitializer

        with tempfile.TemporaryDirectory() as directory:
            transformer = Path(directory) / "transformer"
            transformer.mkdir()
            index = transformer / "model.safetensors.index.json"
            image = Image.new("RGB", (16, 16))

            for level, expected in [(None, None), ("None", None), ("null", None), ("", None), ("4", 4)]:
                with self.subTest(level=level):
                    index.write_text(json.dumps({"metadata": {"quantization_level": level}}))

                    def generate(model, **options):
                        self.assertEqual(model.bits, expected)
                        return SimpleNamespace(image=image)

                    with (
                        patch.object(ZImageInitializer, "_init_tokenizers"),
                        patch.object(ZImageInitializer, "_init_models"),
                        patch.object(ZImage, "generate_image", generate),
                    ):
                        result = generate_z_image("A red teapot", directory, width=16, height=16)
                    self.assertEqual(result.image.size, (16, 16))
