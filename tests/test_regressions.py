import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from ir_dde import enhance_frame, get_preset
from ir_dde.cli.evaluate import evaluate_folder, find_raw_matches
from ir_dde.cli.enhance import build_config
from ir_dde.cli.visualize import build_panel
from ir_dde.legacy import enhance_frame_legacy


class RegressionTests(unittest.TestCase):
    def test_cli_rejects_ignored_overrides(self):
        args = Namespace(preset="natural", use_dog=False, **{key: None for key in (
            "d2br", "detail_gain_max", "detail_amplitude", "detail_threshold_scale",
            "spatial_threshold", "base_local_contrast_mix", "hotspot_protect",
            "output_p_lo", "output_p_hi", "base_method", "plateau_ratio",
        )})
        self.assertTrue(build_config(args).natural_mode)
        args.d2br = 0.0
        self.assertEqual(build_config(args).d2br, 0)
        args.detail_amplitude = 0.9
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_config(args)

    def test_nested_matching_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side, ext in (("raw", ".tiff"), ("out", ".png")):
                for scene in ("a", "b"):
                    folder = root / side / scene
                    folder.mkdir(parents=True)
                    cv2.imwrite(str(folder / ("frame" + ext)), np.zeros((8, 8), np.uint16 if side == "raw" else np.uint8))
            pairs = find_raw_matches(root / "raw", root / "out")
            self.assertEqual(len(pairs), 2)
            for raw, enhanced in pairs:
                self.assertEqual(raw.parent.name, enhanced.parent.name)
            rows = evaluate_folder(root / "raw", root / "out")
            self.assertEqual([r["name"] for r in rows], ["a/frame.tiff", "b/frame.tiff"])
            cv2.imwrite(str(root / "out/a/frame.jpg"), np.zeros((8, 8), np.uint8))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                find_raw_matches(root / "raw", root / "out")

    def test_missing_and_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "out").mkdir()
            cv2.imwrite(str(root / "raw/frame.tiff"), np.zeros((8, 8), np.uint16))
            with self.assertRaises(FileNotFoundError):
                find_raw_matches(root / "raw", root / "out")
            cv2.imwrite(str(root / "out/frame.png"), np.zeros((4, 4), np.uint8))
            with self.assertRaisesRegex(ValueError, "shapes"):
                evaluate_folder(root / "raw", root / "out")

    def test_legacy_input_and_small_histogram(self):
        with self.assertRaisesRegex(ValueError, "uint16"):
            enhance_frame(np.zeros((16, 16), np.float32), get_preset("legacy"))
        raw = np.full((16, 16), 10000, np.uint16)
        raw[:, 8:12] = 10010
        raw[:, 12:] = 10100
        out = enhance_frame_legacy(raw, detail_max=0)
        self.assertGreater(float(out[:, 8:12].mean()), 80)

    def test_debug_panels_and_output_consistency(self):
        raw = np.tile(np.arange(64, dtype=np.uint16), (64, 1)) + 10000
        for name in ("legacy", "natural", "balanced"):
            with self.subTest(preset=name):
                cfg = get_preset(name)
                output, debug = enhance_frame(raw, cfg, True)
                np.testing.assert_array_equal(output, enhance_frame(raw, cfg))
                panel = build_panel(raw, output, debug)
                self.assertEqual(panel.shape, (576, 1440, 3))
                self.assertGreater(panel.std(), 0)

    def test_natural_step_has_no_visible_halo(self):
        raw = np.full((128, 256), 10000, np.uint16)
        raw[:, 128:] = 11000
        output = enhance_frame(raw, get_preset("natural")).astype(float)
        profile = output[64]
        self.assertGreaterEqual(np.diff(profile).min(), -1)
        self.assertLessEqual(np.ptp(profile[10:120]), 1)
        self.assertLessEqual(np.ptp(profile[136:246]), 1)

    def test_natural_scale_invariance_and_finite_input(self):
        rng = np.random.default_rng(7)
        raw = rng.integers(10000, 12000, (128, 128), dtype=np.uint16)
        cfg = get_preset("natural")
        a = enhance_frame(raw, cfg).astype(float)
        b = enhance_frame((raw.astype(np.float32) - 10000) / 2000, cfg).astype(float)
        self.assertLess(np.abs(a - b).mean(), 0.05)
        with self.assertRaisesRegex(ValueError, "finite"):
            enhance_frame(np.full((4, 4), np.nan), cfg)

    def test_natural_noisy_weak_target(self):
        clean = np.full((256, 320), 10000.0)
        clean[:, 160:] = 11000
        clean[96:128, 64:96] += 50
        clean[30:40, 250:260] = 18000
        scores = {"natural": [], "balanced": []}
        for seed in range(5):
            raw = (clean + np.random.default_rng(seed).normal(0, 10, clean.shape)).astype(np.uint16)
            for name in scores:
                out = enhance_frame(raw, get_preset(name)).astype(float)
                target, bg = out[100:124, 68:92], out[100:124, 108:132]
                scores[name].append((target.mean() - bg.mean()) / max(np.sqrt((target.var() + bg.var()) / 2), 1e-6))
        self.assertGreater(np.mean(scores["natural"]), np.mean(scores["balanced"]))


if __name__ == "__main__":
    unittest.main()
