import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from ir_dde import OpenDDEV3Config, enhance_frame, enhance_image_file


class PipelineTest(unittest.TestCase):
    def test_enhance_frame_returns_u8_image(self) -> None:
        image = cv2.imread(str(ROOT / "examples" / "single" / "original_16bit.tif"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)

        output = enhance_frame(image, OpenDDEV3Config())

        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        self.assertGreater(int(output.max()), int(output.min()))

    def test_enhance_image_file_writes_output(self) -> None:
        output_path = Path("/tmp/infrared_dde_test.png")
        metrics = enhance_image_file(ROOT / "examples" / "single" / "original_16bit.tif", output_path, OpenDDEV3Config())

        self.assertTrue(output_path.exists())
        self.assertIn("scene_gain", metrics)
        self.assertGreater(metrics["noise_sigma"], 0.0)


if __name__ == "__main__":
    unittest.main()
