import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'track.py'), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                              env={**os.environ, 'PYTHONUTF8': '1'}, timeout=180)

    def test_missing_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results'
            result = self.run_cli('--source', Path(folder) / 'missing.mp4', '--output', output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Source does not exist', result.stderr)
            self.assertFalse(output.exists())

    def test_blank_video_is_saved_without_fake_tracks(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'blank.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            output = Path(folder) / 'results'
            result = self.run_cli('--source', source, '--output', output, '--device', 'cpu', '--imgsz', 320)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            stats = json.loads((output / 'summary.json').read_text())
            self.assertEqual(stats['frames'], 4)
            self.assertEqual(stats['unique_track_ids'], 0)
            with (output / 'tracks.csv').open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            cap = cv2.VideoCapture(str(output / 'tracked.mp4'))
            count = 0
            while cap.read()[0]:
                count += 1
            cap.release()
            self.assertEqual(count, 4)
            # Refuse reuse rather than silently overwriting a previous run.
            repeated = self.run_cli('--source', source, '--output', output)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('already exists', repeated.stderr)
