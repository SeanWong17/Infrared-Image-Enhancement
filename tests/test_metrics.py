import unittest
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

from ir_dde import evaluate_against_linear_baseline, evaluate_display_image


class MetricsTest(unittest.TestCase):
    def test_display_metrics_are_positive(self) -> None:
        image = cv2.imread(str(ROOT / "docs" / "assets" / "enhanced_8bit.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)

        metrics = evaluate_display_image(image)

        self.assertGreater(metrics["entropy"], 0.0)
        self.assertGreater(metrics["avg_gradient"], 0.0)
        self.assertGreater(metrics["eme"], 0.0)

    def test_relative_metrics_include_gain_terms(self) -> None:
        raw = cv2.imread(str(ROOT / "examples" / "single" / "original_16bit.tif"), cv2.IMREAD_UNCHANGED)
        enhanced = cv2.imread(str(ROOT / "docs" / "assets" / "enhanced_8bit.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(raw)
        self.assertIsNotNone(enhanced)

        metrics = evaluate_against_linear_baseline(raw, enhanced)

        self.assertIn("entropy_gain", metrics)
        self.assertIn("eme_gain", metrics)


if __name__ == "__main__":
    unittest.main()
