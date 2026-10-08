"""数据集读取模块。



设计目标：

- 保证严格按 fold 划分 train/val/test

- 支持 HDF5 与 memmap 双读取路径

- 为预训练提供三元组样本，为微调提供序列样本

"""

from __future__ import annotations



from dataclasses import dataclass

from pathlib import Path

from typing import Optional



import h5py

import numpy as np

import torch

from torch.utils.data import Dataset





@dataclass

class H5Keys:

    rhythms: str = "x_rhythm"

    labels: str = "y"

    subjects: str = "subject_id"

    split_prefix: str = "split_fold_"

    seq_index: str = "seq_index"

    triplet_index: str = "triplet_index"





class _BaseH5Dataset(Dataset):

    def __init__(

        self,

        h5_path: str,

        split_name: str,

        fold: int = 0,

        keys: H5Keys | None = None,

        memmap_dir: str | None = None,

    ) -> None:

        self.h5_path = h5_path

        self.split_name = split_name

        self.fold = int(fold)

        self.keys = keys if keys is not None else H5Keys()

        self.memmap_dir = Path(memmap_dir).expanduser().resolve() if memmap_dir else None

        self._h5: Optional[h5py.File] = None

        self._x_mm: Optional[np.ndarray] = None

        self._y_mm: Optional[np.ndarray] = None



        self._x_memmap_path: Optional[Path] = None

        self._y_memmap_path: Optional[Path] = None

        if self.memmap_dir is not None:

            x_path = self.memmap_dir / f"{self.keys.rhythms}.npy"

            y_path = self.memmap_dir / f"{self.keys.labels}.npy"

            self._x_memmap_path = x_path if x_path.exists() else None

            self._y_memmap_path = y_path if y_path.exists() else None



        with h5py.File(self.h5_path, "r") as f:

            split_key = f"{self.keys.split_prefix}{self.fold}"

            if split_key in f:

                split_raw = np.asarray(f[split_key])

                if split_raw.dtype.kind in {"S", "O", "U"}:

                    split_text = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in split_raw])

                    self.base_indices = np.where(split_text == split_name)[0].astype(np.int64)

                else:

                    # numeric split coding: 0=train,1=val,2=test

                    code = {"train": 0, "val": 1, "test": 2}[split_name]

                    self.base_indices = np.where(split_raw == code)[0].astype(np.int64)

            else:

                raise KeyError(

                    f"Missing split key '{split_key}' in HDF5. "

                    "Please run scripts/make_subject_split.py to generate fold splits."

                )



    @property

    def using_memmap(self) -> bool:

        return self._x_memmap_path is not None



    def _ensure_open(self) -> h5py.File:

        if self._h5 is None:

            # Per-worker chunk cache helps repeated random reads from compressed HDF5.

            self._h5 = h5py.File(

                self.h5_path,

                "r",

                rdcc_nbytes=64 * 1024 * 1024,

                rdcc_nslots=100_003,

                rdcc_w0=0.75,

            )

        return self._h5



    def _ensure_x_array(self):

        if self.using_memmap:

            if self._x_mm is None:

                self._x_mm = np.load(self._x_memmap_path, mmap_mode="r")

            return self._x_mm

        f = self._ensure_open()

        return f[self.keys.rhythms]



    def _ensure_y_array(self):

        if self._y_memmap_path is not None:

            if self._y_mm is None:

                self._y_mm = np.load(self._y_memmap_path, mmap_mode="r")

            return self._y_mm

        f = self._ensure_open()

        return f[self.keys.labels]





class H5PretrainDataset(_BaseH5Dataset):

    """Pretraining dataset with temporal neighbors (t-1, t, t+1).



    Output item:

    - rhythm_t: (5, 3000)

    - rhythm_prev: (5, 3000)

    - rhythm_next: (5, 3000)

    - has_prev: bool

    - has_next: bool

    """



    def __init__(

        self,

        h5_path: str,

        split_name: str = "train",

        fold: int = 0,

        keys: H5Keys | None = None,

        memmap_dir: str | None = None,

    ) -> None:

        super().__init__(h5_path, split_name, fold=fold, keys=keys, memmap_dir=memmap_dir)

        with h5py.File(self.h5_path, "r") as f:

            # Enforce precomputed triplets to avoid invalid temporal neighbors.

            if self.keys.triplet_index not in f:

                raise FileNotFoundError(

                    f"Pretraining requires '{self.keys.triplet_index}' dataset in HDF5. "

                    "Please run scripts/build_triplet_index.py first to generate it."

                )



            trip = np.asarray(f[self.keys.triplet_index], dtype=np.int64)

            split_set = set(self.base_indices.tolist())

            keep_mask = np.array([int(triplet[1]) in split_set for triplet in trip], dtype=bool)

            self.triplet = trip[keep_mask]

            if len(self.triplet) == 0:

                raise ValueError(f"No valid triplets found for split {split_name}, check your triplet generation.")



            # Rigor guard: each (prev, t, next) triplet must belong to one subject.

            # Prevents cross-subject neighbors from corrupting the temporal-smoothing

            # objective during self-supervised pretraining.

            trip_subjects = np.asarray(f[self.keys.subjects], dtype=np.int64)[self.triplet]

            triplet_same_subject = (trip_subjects == trip_subjects[:, :1]).all(axis=1)

            if not triplet_same_subject.all():

                bad = int((~triplet_same_subject).sum())

                first_bad = int(np.flatnonzero(~triplet_same_subject)[0])

                raise AssertionError(

                    f"[H5PretrainDataset] {bad} triplet(s) span multiple subjects "

                    f"(e.g. row {first_bad}: subjects={trip_subjects[first_bad].tolist()}). "

                    "Rebuild triplet_index with scripts/build_triplet_index.py."

                )



        if self.memmap_dir is not None and not self.using_memmap:

            raise FileNotFoundError(

                f"Memmap requested but missing file: {self.memmap_dir / f'{self.keys.rhythms}.npy'}"

            )



    def __len__(self) -> int:

        return len(self.triplet)



    def __getitem__(self, idx: int) -> dict:

        x = self._ensure_x_array()



        # Triplet indices are precomputed to ensure temporal adjacency.

        i_prev, i_t, i_next = map(int, self.triplet[idx])



        n = x.shape[0]

        has_prev = 0 <= i_prev < n

        has_next = 0 <= i_next < n



        rhythm_t = np.asarray(x[i_t], dtype=np.float32)

        rhythm_prev = np.asarray(x[i_prev], dtype=np.float32) if has_prev else np.zeros_like(rhythm_t, dtype=np.float32)

        rhythm_next = np.asarray(x[i_next], dtype=np.float32) if has_next else np.zeros_like(rhythm_t, dtype=np.float32)



        return {

            "rhythm_t": torch.from_numpy(rhythm_t),

            "rhythm_prev": torch.from_numpy(rhythm_prev),

            "rhythm_next": torch.from_numpy(rhythm_next),

            "has_prev": torch.tensor(has_prev, dtype=torch.bool),

            "has_next": torch.tensor(has_next, dtype=torch.bool),

        }





class H5FinetuneDataset(_BaseH5Dataset):

    """Fine-tuning sequence dataset.



    Sequence label is the label of the last epoch in sequence.

    """



    def __init__(

        self,

        h5_path: str,

        split_name: str = "train",

        fold: int = 0,

        keys: H5Keys | None = None,

        seq_len: int = 20,

        memmap_dir: str | None = None,

    ) -> None:

        super().__init__(h5_path, split_name, fold=fold, keys=keys, memmap_dir=memmap_dir)

        self.seq_len = seq_len



        with h5py.File(self.h5_path, "r") as f:

            subject_ids = np.asarray(f[self.keys.subjects], dtype=np.int64)

            if self.keys.seq_index in f:

                seq = np.asarray(f[self.keys.seq_index], dtype=np.int64)

            else:

                seq = self._build_fallback_seq_index(subject_ids, seq_len)



            split_set = set(self.base_indices.tolist())

            keep = []

            for i in range(len(seq)):

                if all(int(j) in split_set for j in seq[i]):

                    keep.append(i)

            self.seq_index = seq[np.asarray(keep, dtype=np.int64)] if keep else np.zeros((0, seq_len), dtype=np.int64)



            # Rigor guard: every sequence must come from a single subject.

            # This hard assertion prevents cross-subject artifact sequences from

            # silently entering a batch, which would corrupt temporal modeling.

            if len(self.seq_index) > 0:

                seq_subjects = subject_ids[self.seq_index]  # (num_seq, seq_len)

                same_subject = (seq_subjects == seq_subjects[:, :1]).all(axis=1)

                if not same_subject.all():

                    bad = int((~same_subject).sum())

                    first_bad = int(np.flatnonzero(~same_subject)[0])

                    raise AssertionError(

                        f"[H5FinetuneDataset] {bad} sequence(s) span multiple subjects "

                        f"(e.g. row {first_bad}: subjects={seq_subjects[first_bad].tolist()}). "

                        "Rebuild seq_index with scripts/build_seq_index.py."

                    )



        if self.memmap_dir is not None:

            missing = []

            if not self.using_memmap:

                missing.append(str(self.memmap_dir / f"{self.keys.rhythms}.npy"))

            if self._y_memmap_path is None:

                missing.append(str(self.memmap_dir / f"{self.keys.labels}.npy"))

            if missing:

                raise FileNotFoundError("Memmap requested but missing file(s): " + ", ".join(missing))



    @staticmethod

    def _build_fallback_seq_index(subject_ids: np.ndarray, seq_len: int) -> np.ndarray:

        seq = []

        n = len(subject_ids)

        for i in range(0, n - seq_len + 1):

            sub = subject_ids[i : i + seq_len]

            if np.all(sub == sub[0]):

                seq.append(np.arange(i, i + seq_len, dtype=np.int64))

        if not seq:

            return np.zeros((0, seq_len), dtype=np.int64)

        return np.stack(seq, axis=0)



    def __len__(self) -> int:

        return len(self.seq_index)



    def __getitem__(self, idx: int) -> dict:

        x = self._ensure_x_array()

        y = self._ensure_y_array()



        ids = self.seq_index[idx]

        seq_rhythm = np.asarray(x[ids], dtype=np.float32)  # (seq_len, 5, 3000)

        label = int(y[int(ids[-1])])

        return {

            "seq_rhythm": torch.from_numpy(seq_rhythm),

            "label": torch.tensor(label, dtype=torch.long),

            "index_last": torch.tensor(int(ids[-1]), dtype=torch.long),

        }

