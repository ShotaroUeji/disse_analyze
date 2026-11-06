from pathlib import Path
import unittest

import numpy as np
import torch

from cross_source_geometry.cross_source_distance import pair_distance
from cross_source_geometry.cross_source_retrieval import analyze_records
from cross_source_geometry.load_embeddings import EmbeddingRecord, load_embedding
from cross_source_geometry.procrustes_analysis import pair_procrustes
from cross_source_geometry.rsa_analysis import rsa_pair


ANGLES = np.arange(0.0, 360.0, 5.0)


def _base_embeddings() -> torch.Tensor:
    radians = np.deg2rad(ANGLES)
    values = np.stack([np.cos(radians), np.sin(radians), np.ones_like(radians), np.cos(2 * radians)], axis=1)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return torch.tensor(values, dtype=torch.float32)


def _record(source_id: str, embeddings: torch.Tensor) -> EmbeddingRecord:
    return EmbeddingRecord(
        source_id=source_id,
        model="disse",
        axis="azimuth",
        room="medium",
        t30_s=0.3,
        condition="medium_t30_0p3s",
        angles_deg=ANGLES.copy(),
        embeddings=embeddings,
        path=Path(f"{source_id}.pt"),
        embedding_key="audio_space",
        source_norm_min=1.0,
        source_norm_max=1.0,
    )


class CrossSourceGeometryTests(unittest.TestCase):
  def test_source_invariant_distance_and_retrieval(self):
    a = _record("a", _base_embeddings())
    b = _record("b", _base_embeddings())
    pair = pair_distance(a, b)
    self.assertAlmostEqual(pair.same_angle_mean, 0.0, places=6)
    self.assertAlmostEqual(rsa_pair(a, b), 1.0, places=6)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
      result = analyze_records([a, b], Path(directory))
      summary = result["summary"].set_index("task")
      self.assertAlmostEqual(float(summary.loc["pairwise_cross_source", "recall_at_1"]), 1.0, places=6)


  def test_orthogonal_rotation_preserves_rsa_and_is_correctable(self):
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a = _record("a", _base_embeddings())
    b = _record("b", _base_embeddings() @ torch.tensor(q, dtype=torch.float32))
    self.assertAlmostEqual(rsa_pair(a, b), 1.0, places=5)
    result = pair_procrustes(a, b)
    self.assertLess(result["test_distance_after"], result["test_distance_before"])


  def test_non_linear_geometry_lowers_rsa(self):
    radians = np.deg2rad(ANGLES)
    values = np.stack([np.cos(2 * radians), np.sin(2 * radians), np.ones_like(radians), np.cos(4 * radians)], axis=1)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    self.assertLess(rsa_pair(_record("a", _base_embeddings()), _record("b", torch.tensor(values, dtype=torch.float32))), 0.9)


  def test_azimuth_wraparound_delta(self):
    pair = pair_distance(_record("a", _base_embeddings()), _record("b", _base_embeddings()))
    self.assertIn(5.0, pair.delta_means)


  def test_loader_rejects_nan(self):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp) / "source" / "disse" / "azimuth" / "medium_t30_0p3s"
      directory.mkdir(parents=True)
      torch.save({"angles_deg": torch.tensor(ANGLES), "embeddings": {"audio_space": torch.full((72, 4), float("nan"))}}, directory / "continuous_azimuth_embeddings.pt")
      with self.assertRaisesRegex(ValueError, "NaN"):
        load_embedding(directory / "continuous_azimuth_embeddings.pt")


if __name__ == "__main__":
  unittest.main()
