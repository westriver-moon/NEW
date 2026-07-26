from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


class AutoReplaceIdentityBatchSampler(Sampler[List[int]]):
    def __init__(self, pid_to_indices: Dict[int, Sequence[int]], batch_size: int, instances_per_identity: int, seed: int = 0):
        if batch_size % instances_per_identity:
            raise ValueError("batch_size must be divisible by instances_per_identity")
        self.pid_to_indices = {int(pid): list(indices) for pid, indices in pid_to_indices.items()}
        self.batch_size = int(batch_size)
        self.instances = int(instances_per_identity)
        self.identities_per_batch = self.batch_size // self.instances
        self.seed = int(seed)
        self.epoch = 0
        if len(self.pid_to_indices) < self.identities_per_batch:
            raise ValueError("not enough identities for one batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, sum(len(v) for v in self.pid_to_indices.values()) // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        pids = list(self.pid_to_indices)
        rng.shuffle(pids)
        cursor = 0
        for _ in range(len(self)):
            if cursor + self.identities_per_batch > len(pids):
                rng.shuffle(pids)
                cursor = 0
            selected = pids[cursor : cursor + self.identities_per_batch]
            cursor += self.identities_per_batch
            batch: List[int] = []
            for pid in selected:
                candidates = self.pid_to_indices[pid]
                if len(candidates) >= self.instances:
                    batch.extend(rng.sample(candidates, self.instances))
                else:
                    batch.extend(rng.choice(candidates) for _ in range(self.instances))
            yield batch

