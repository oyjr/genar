from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import safe_io


class UnsafePayload:
    pass


class SafeIOTests(unittest.TestCase):
    def test_tensor_only_payload_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tensor.pt'
            torch.save({'value': torch.arange(3)}, path)
            loaded = safe_io.safe_torch_load(path)
        self.assertTrue(torch.equal(loaded['value'], torch.arange(3)))

    def test_pickle_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'object.pt'
            torch.save(UnsafePayload(), path)
            with self.assertRaisesRegex(ValueError, 'tensor-only mode'):
                safe_io.safe_torch_load(path)

    def test_public_filename_removes_posix_and_windows_parents(self):
        self.assertEqual(
            safe_io.public_filename('/example/checkpoints/model.ckpt'),
            'model.ckpt',
        )
        self.assertEqual(
            safe_io.public_filename(
                PureWindowsPath(r'C:\example\checkpoints\model.ckpt')
            ),
            'model.ckpt',
        )

    def test_vulnerable_torch_version_is_rejected(self):
        with (
            mock.patch.object(safe_io.torch, '__version__', '2.5.1'),
            self.assertRaisesRegex(RuntimeError, '2.6.0 or newer'),
        ):
            safe_io.require_safe_torch_load()


if __name__ == '__main__':
    unittest.main()
