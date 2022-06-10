import torch
from torch import nn
import torch.optim
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

CURRENT_DATA_FILE = os.path.join(
    "..",
    "VENUS_data_folder",
    "VENUS_current_data.csv"
)

class VENUS_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, usecols_input, usecols_output, device=device):
        self.input_data = data[usecols_input]
        self.output_data = data[usecols_output]
        self.device = device

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.input_data.iloc[idx],
                device=self.device, dtype=torch.float32),
            "output": torch.tensor(self.output_data.iloc[idx],
                device=self.device, dtype=torch.float32)
        }

def read_and_split_data(val_proportion, trial=None):
    data = pd.read_csv(CURRENT_DATA_FILE)
    if trial is not None:
        data = data[data["trial index"].isin(trial)]
    data = data.sample(frac=1)
    split_idx = int(val_proportion * len(data))
    return data[split_idx:], data[:split_idx]

def read_and_split_data_by_trial(train_idx=None, val_idx=None):
    assert train_idx is not None or val_idx is not None
    if train_idx is not None:
        train_idx = set(train_idx)
    if val_idx is not None:
        val_idx = set(val_idx)
    data = pd.read_csv(CURRENT_DATA_FILE)
    idx = set(list(range(data["trial index"].max())))
    if train_idx is None:
        train_idx = idx - val_idx
    elif val_idx is None:
        val_idx = idx - train_idx
    train_idx = data["trial index"].isin(train_idx)
    val_idx =  data["trial index"].isin(val_idx)
    return data[train_idx], data[val_idx]
