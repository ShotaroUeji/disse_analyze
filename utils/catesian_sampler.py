from __future__ import annotations
from typing import List, Dict, Tuple
from torch.utils.data import Sampler
from collections import defaultdict
import torch
import pandas as pd

class CartesianBatchSampler(Sampler[List[int]]):
    """
    Validation を train の構成 (dry_per_batch × rirs_per_batch) に合わせて
    (audiocap_id, rir_path) の直積で 1 バッチを作るサンプラ。

    - 各バッチ: dry_per_batch 個の audio と rirs_per_batch 個の RIR を選び、
      その直積 (dry_per_batch × rirs_per_batch) のインデックスを列挙。
    - 例: dry_per_batch=24, rirs_per_batch=3 → バッチサイズ=72

    NOTE:
      * CSV が (audiocap_id, rir_path) の全組合せを含む前提。
      * 再現性重視のためデフォルトは非シャッフル。
    """
    def __init__(
        self,
        index_csv: str,
        dry_per_batch: int = 24,
        rirs_per_batch: int = 3,
        drop_last: bool = True,
        shuffle_audio: bool = False,
        shuffle_rir: bool = False,
        seed: int = 0,
    ):
        assert dry_per_batch >= 1 and rirs_per_batch >= 1
        self.dry_per_batch = dry_per_batch
        self.rirs_per_batch = rirs_per_batch
        self.drop_last = drop_last
        self.shuffle_audio = shuffle_audio
        self.shuffle_rir = shuffle_rir
        self.seed = seed

        df = pd.read_csv(index_csv)
        # 一意リスト（決定的順序: 出現順 or ソート）
        self._audios = df["audiocap_id"].drop_duplicates().tolist()
        self._rirs   = df["rir_path"].drop_duplicates().tolist()

        # (audio, rir) → row_index のマップを作る
        # 同一ペアが複数行あるケースは想定外だが、先に出た1行を採用
        pair2idx: Dict[Tuple[int, str], int] = {}
        for i, row in df.iterrows():
            k = (int(row["audiocap_id"]), row["rir_path"])
            if k not in pair2idx:
                pair2idx[k] = int(i)
        self._pair2idx = pair2idx

        # 直積が全部存在するかを（使う範囲で）逐次チェックするので、ここでは assert しない

    def __len__(self) -> int:
        n_audio_blocks = len(self._audios) // self.dry_per_batch
        n_rir_blocks   = len(self._rirs)   // self.rirs_per_batch
        if self.drop_last:
            return n_audio_blocks * n_rir_blocks
        # 端数も 1 ブロックとして数える（可変バッチを許容する場合）
        rem_a = 1 if (len(self._audios) % self.dry_per_batch) else 0
        rem_r = 1 if (len(self._rirs)   % self.rirs_per_batch) else 0
        return (n_audio_blocks + rem_a) * (n_rir_blocks + rem_r)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)

        audios = list(self._audios)
        rirs   = list(self._rirs)
        if self.shuffle_audio:
            audios = [audios[i] for i in torch.randperm(len(audios), generator=g).tolist()]
        if self.shuffle_rir:
            rirs = [rirs[i] for i in torch.randperm(len(rirs), generator=g).tolist()]

        def chunks(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i+size]

        audio_blocks = list(chunks(audios, self.dry_per_batch))
        rir_blocks   = list(chunks(rirs,   self.rirs_per_batch))

        if self.drop_last:
            audio_blocks = [b for b in audio_blocks if len(b) == self.dry_per_batch]
            rir_blocks   = [b for b in rir_blocks   if len(b) == self.rirs_per_batch]

        for a_block in audio_blocks:
            for r_block in rir_blocks:
                batch = []
                for a in a_block:
                    for r in r_block:
                        k = (int(a), r)
                        if k not in self._pair2idx:
                            # 直積が欠けている場合はスキップ（想定外だが安全側）
                            continue
                        batch.append(self._pair2idx[k])
                # 可変長を避けたい場合は drop する
                if len(batch) == self.dry_per_batch * self.rirs_per_batch:
                    yield batch
                elif not self.drop_last and len(batch) > 0:
                    yield batch
