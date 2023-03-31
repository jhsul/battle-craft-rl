from typing import List
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class Memory:
    """
    This class represents a single frame/step of the agent
    A full episode should be of type List[Memory]
    """
    # obs: np.ndarray
    # state: np.ndarray
    pi_h: np.ndarray
    v_h: np.ndarray
    action: np.ndarray
    action_log_prob: np.ndarray
    reward: float
    total_reward: float
    done: bool
    value: float


class MemoryDataset(Dataset):
    """
    This is a dataset of memory objects (potentially multiple episodes!)
    This is to be used with the PyTorch DataLoader
    """

    def __init__(self, memories: List[Memory]):
        self.memories: List[Memory] = memories

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, idx):
        mem = self.memories[idx]

        # This needs to be returned as a tuple
        return mem.pi_h, mem.v_h, mem.action, mem.action_log_prob, mem.reward, mem.total_reward, mem.done, mem.value


# This is probably not needed, but might as well define this type so we have it
# Edit: this is almost certainly not useful since we shuffle memories anyway
Episode = List[Memory]
