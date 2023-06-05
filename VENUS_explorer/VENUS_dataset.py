"""
The VENUS dataset. 
    Date: May 2023
    Author: Wenhan Sun
"""

import torch
from torch import nn
import torch.optim
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import interval
from typing import Optional

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

"""
The current VENUS data. 
"""
CURRENT_DATA_FILE = os.path.join(
    "..",
    "VENUS_data_folder",
    "VENUS_current_data.csv"
)

def is_iterable(obj):
    """
    Return true if an object is an iterable.
    @params obj: The object to be tested.
    @return (bool): True iff the object is an iterable. 
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False

class Naive_VENUS_Dataset(torch.utils.data.Dataset):
    """
    The VENUS dataset. Data is directly provided in the form of a dataframe. 

    @field input_data (pandas.DataFrame): The input dataframe.
    @field output_data (pandas.DataFrame): The output dataframe. 
    @field device (torch.device): The deviced used to store the data. 
    """
    def __init__(self, data: pd.DataFrame, usecols_input: list, 
            usecols_output: list, device: torch.device = device):
        """
        Constructor.
        @params data (pandas.DataFrame): The data.
        @params usecols_input (list): A list of columns used as input.
        @params usecols_output (list): A list of columns used as output.
        @params device (torch.device): The deviced used to store the data. 
        """
        self.input_data = data[usecols_input]
        self.output_data = data[usecols_output]
        self.device = device

    def __len__(self):
        """
        OVERRIDE
        """
        return len(self.input_data)

    def __getitem__(self, idx):
        """
        OVERRIDE
        @return (dict): A dictionary with input and output entries.
        """
        return {
            "input": torch.tensor(self.input_data.iloc[idx],
                device=self.device, dtype=torch.float32),
            "output": torch.tensor(self.output_data.iloc[idx],
                device=self.device, dtype=torch.float32)
        }

class VENUS_h5_Dataset(torch.utils.data.Dataset):
    """
    Deprecated!
    """
    def __init__(self, filename, vars="full", thres=5e-9, device=device):
        raise DeprecationWarning("This dataset is deprecated!")
        super(VENUS_h5_Dataset, self).__init__()
        self.f = h5py.File(filename, "r")
        self.device = device
        if vars == "full":
            self.get_full_vars("single")
        elif vars == "simple":
            self.get_simple_vars()
        elif vars is not None:
            self.input_vars = vars["input"]
            self.output_vars = vars["output"]
        self.indices = np.nonzero(self.f["faraday_cup"][()]['fcv1_i'] > thres)[0]

    def get_full_vars(self, output="single"):
        keys = self.f.keys()
        self.input_vars = []
        for key in keys:
            if keys != "faraday_cup":
                length = len(self.f[key][0]) - 1
                self.input_vars.append((key, range(length)))
        if output == "single":
            self.output_vars = [("faraday_cup", [1])]
        else:
            raise ValueError(f"Output mode not acceptable: {output}")

    def get_simple_vars(self):
        self.input_vars = [
            ("superconductor", [0, 1, 2]),
            ("bias", [1]),
            ("vacuum", [1, 0])
        ]
        self.output_vars = [("faraday_cup", [1])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        i = self.indices[i]
        input = [self.f[group[0]][i][idx] for group in self.input_vars for idx in group[1]]
        output = [self.f[group[0]][i][idx] for group in self.output_vars for idx in group[1]]
        return {
            "input": torch.tensor(input,
                device=self.device, dtype=torch.float32),
            "output": torch.tensor(output,
                device=self.device, dtype=torch.float32)
        }

    def print_fields(self):
        for key in self.f.keys():
            print(key, self.f[key][()].dtype.names)

class VENUS_Dataset(torch.utils.data.Dataset):
    """
    The full VENUS dataset from a folder.
    
    @field data (pandas.DataFrame): The data
    @field device (torch.device): The device where the parameters are stored. 
    @field cols_dict (dict): The columns dictionary, whose values must be 
        columns of `data`. `"input"` and `"output"` columns must be specified. 
        cols_dict["input"]: Input features.
        cols_dict["output"]: Output features.
    """
    def __init__(self, filename: str, 
            times: Optional[interval.Intervals] = None, 
            device: torch.device = device, 
            drop_na: bool = False, 
            csv_kwargs: dict = {}, 
            data: Optional[pd.DataFrame] = None):
        """
        Constructor.
        @params filename (str): The file name where the data is located. If 
            it is an empty string, then the data is provided explicitly. 
            Supported file types include h5 and csv files. 
        @params device (torch.device): The device where the parameters are 
            stored. 
        @params drop_na (bool, DEFAULT False): True iff the NAN in the data 
            are dropped row-wise.
        @params csv_kwargs (dict): The argument provided to `pandas.read_csv` 
            when the data is loaded. 
        @params data (Optional[pandas.DataFrame], DEFAULT None): The manually 
            entered data. Must be specified when no file name is given. 
            CONSTRAINT: filename or data
        @error (ValuError): Error if neither file name nor data are given. 
        """
        super(VENUS_Dataset, self).__init__()
        if filename:
            self.load_data(filename, csv_kwargs=csv_kwargs)
            if drop_na:
                self.data = self.data.dropna()
            if times:
                self.data = self.data[self.data["unix_epoche_milliseconds"].map(times.isin)]
        else:
            if data is None:
                raise ValueError("Either file name or data must be provided!")
            self.data = data.copy()
        self.device = device
        self.cols_dict = {}

    def load_data(self, filename: str, csv_kwargs: dict = {}) -> None:
        """
        Load the data from file.
        @params filename (str): The file name where the data is located. 
            Supported file types include h5 and csv files. 
        @params csv_kwargs (dict): The argument provided to `pandas.read_csv` 
            when the data is loaded. 
        @error (ValueError): Error if the file is not supported. 
        """
        if filename[-3:] == ".h5":
            # Load from h5 files. "unix_epoche_millisecond" must be a feature. 
            self.f = h5py.File(filename, "r")
            keys = list(self.f.keys())
            self.data = [(key, self.f[key][()]) for key in keys]
            self.data = [(key,
                pd.DataFrame(df).sort_values(by=["unix_epoche_milliseconds"]).reset_index(drop=True)
                ) for key, df in self.data]
            self.data = [
                (key, df.drop("unix_epoche_milliseconds", axis=1))
                if key != "faraday_cup" else (key, df)
                for key, df in self.data
            ]
            self.data = [df.rename(lambda x: f"{key} {x}", axis="columns")
                for key, df in self.data]
            self.data = pd.concat(self.data, axis=1)
            self.data = self.data.rename(
                columns=
                {x: x.split()[-1] for x in self.data.columns}
            )
        elif filename[-4:] == ".csv":
            # Load from csv files. 
            self.data = pd.read_csv(filename, **csv_kwargs)
            def rename_fn(col):
                # Rename the columns recursively handling multi-index. 
                if isinstance(col, str):
                    return "" if "Unnamed" in col else col
                else:
                    return (rename_fn(c) for c in col)
            self.data = self.data.rename(columns=rename_fn)
        else:
            raise ValueError(f"{filename} cannot be loaded as dataset!")

    def filter_value(self, col, value, mode: str = "=", 
            rel_tolerance: float = 0.01) -> None:
        """
        Filter the dataset according to some criterion.
        @params col: The column or columns on which the data is filtered. 
            If it is a list, then multiple columns are filtered simultaneous, 
            linked by and. If it is `"__ALL__"`, then all the columns are 
            used. Otherwise, it represents a single column. 
            CONSTRAINT: not isinstance(col, list) or \
                [c in self.data.columns for c in col]
            CONSTRAINT: isinstance(col, list) or (col in self.data.columns) \
                    or (col == "__ALL__")
        @params value: The value to be compared with. If it is a number, then 
            it is used across all columns. If it is of class `pandas.Series`, 
            then the value is column-wise, whose indices must match `col`. If 
            it is the string `"median"`, then the median of each column is 
            used. If it is the string `"mean"`, the the mean of each column is 
            used.
        @params mode (str, DEFAULT "="): The mode of filter. 
            "=": Equal to a particular value.
            ">": More than a particular value.
            ">=": No less than a particular value.
            "<": Less than a particular value.
            "<=": No more than a particular value.
            "in": Within a particular collection of intervals.
            "approx": Approximately equal to a particular value based on a 
            tolerance. 
            CONSTRAINT: mode in ["=", "<", ">", "<=", ">=", "approx", "in"]
        @params rel_tolerance (float, DEFAULT 0.01): The relative tolerance in 
            the `"approx"` mode. 
            CONSTRAINT: rel_tolereance > 0
        @error (ValueError): Error if the mode is not supported. 
        """
        multi_cols = isinstance(col, list) or col == "__ALL__"
        col_data = self.data[col] if col != "__ALL__" else self.data
        if isinstance(value, str):
            if value == "median":
                value = col_data.median()
            elif value == "mean":
                value = col_data.mean()
            else:
                raise ValueError(f"{value} is not supported as string. ")
        if mode == "=":
            indicator = col_data == value
        elif mode == "in":
            indicator = col_data.isin(value)
        elif mode in [">", ">=", "<", "<="]:
            indicator = eval(f"col_data {mode} value")
        elif mode == "approx":
            low_thres = value * (1 - rel_tolerance)
            high_thres = value * (1 + rel_tolerance)
            low_thres, high_thres = \
                min(low_thres, high_thres), max(low_thres, high_thres)
            indicator = (col_data > low_thres) & (col_data < high_thres)
        else:
            raise ValueError(f"Mode {mode} is not defined!")
        indicator = indicator.all(axis=1) if multi_cols else indicator
        self.data = self.data[indicator]

    def get_statistics(self, col, mode, params = 0) -> pd.Series|float:
        """
        Get the statistics of columns.
        @params col: The column or columns where the statistics is retrieved. 
            If it is a list, then multiple columns are filtered simultaneous, 
            linked by and. If it is `"__ALL__"`, then all the columns are 
            used. Otherwise, it represents a single column. 
            CONSTRAINT: not isinstance(col, list) or \
                [c in self.data.columns for c in col]
            CONSTRAINT: isinstance(col, list) or (col in self.data.columns) \
                    or (col == "__ALL__")
        @params mode (str): The mode of statistics. 
            "quantile": The quantile of the columns. 
            # This is the only mode supported for now. 
            CONSTRAINT: mode in ["quantile"]
        @params params (DEFAULT 0): The parameters to get the statistics. 
            If the quantile is obtained, it is the percentage of the quantile 
            calculated. 
        @return (pd.Series|float): The retrieved statistics of the column or 
            list of columns. 
        @error (ValueError): Error if the mode is not supported. 
        """
        col_data = self.data[col] if col != "__ALL__" else self.data
        if mode == "quantile":
            if params > 1 or params < 0:
                raise ValueError("Quantile mode must have a parameter "
                    "between 0 and 1.")
            result = col_data.quantile(params)
        else:
            raise ValueError(f"Mode {mode} is not defined!")
        return result

    def get_cols(self) -> pd.Index:
        """
        Get the columns of the dataframe.
        @return (pandas.Index): The columns of the dataframe. 
        """
        return self.data.columns

    def set_cols(self, cols_dict: dict) -> None:
        """
        Set the column dictionary of the dataset. 
        @params cols_dict (dict): The columns dictionary, whose values must be 
            columns of `data`. `"input"` and `"output"` columns must be specified. 
            cols_dict["input"]: Input features.
            cols_dict["output"]: Output features.
            CONSTRAINT: "input" in cols_dict.keys() and \
                "output" in cols_dict.keys()
        """
        self.cols_dict = cols_dict
    
    def construct_dataset(self):
        """
        Construct the dataset. Column dictionary must have already been 
            specified.  
        """
        self.tensor_data = {
            key:torch.tensor(self.data[cols].to_numpy(), 
                device=self.device, dtype=torch.float32) 
            for key, cols in self.cols_dict.items()
        }

    def idx_to_item(self, i: int, cols: list) -> torch.Tensor:
        """
        Get an indexed datapoint with given features.
        @params i (int): The integer index of the datapoint.
            CONSTRAINT: i >= 0 and i < len(self.data)
        @params cols (list): The columns to extract.
            CONSTRAINT: all([c in self.data.columns for c in cols])
        @return (torch.Tensor): The return datapoint with features in order 
            with `cols`.
        """
        return torch.tensor(self.data.iloc[i][cols].to_numpy(), 
            device=self.device, dtype=torch.float32)
        
    def __len__(self):
        """
        OVERRIDE
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        OVERRIDE
        @return (dict): The dictionary with entries specified by the column
            dictionary. 
        """
        return {k:v[i] for k,v in self.tensor_data.items()}

    def get_dataframe(self, key) -> pd.DataFrame:
        """
        Get the dataframe of a particular column dictionary key.
        @params key: The key of the columns dictionary. 
        @return (pandas.DataFrame): The dataframe associated with the desired 
            feature given by the column dictionary key. 
        """
        return self.data[self.cols_dict[key]]
    
    def recreate_dataframe(self, data_dict, others):
        """
        Recreate the dataframe based on Pytorch tensors.
        @params data_dict (dict[Any, torch.Tensor]): The dictionary containing 
            key data pair. All the keys should also be the keys of the column 
            dictionary. The values should be of the same order. 
            CONSTRAINT: [k in self.cols_dict.keys() for k in data_dict.keys()] 
        @params other (pd.DataFrame): Additional columns in order with the 
            values in the data dictionary. 
        @return (pandas.DataFrame): The reconstructed dataframe. 
        """
        data = [pd.DataFrame(v, columns=self.cols_dict[k]) 
            for k,v in data_dict.items()]
        return pd.concat(data + [others], axis=1)
    
    def random_split(self, ratios: list[float]) -> list:
        """
        Randomly split the dataset into datasets by a specific ratio. 
        @params ratios (list[float]): The relative ratio of the length of the sub 
            datasets. The ratio does not need to be normalized to 1. 
        @return (list): A list of sub datasets with desired length. 
        """
        datasets = []
        data = self.data.copy()
        while ratios:
            ratios = ratios / ratios.sum()
            sub_data = data.sample(frac=ratios[0])
            data = data.drop(sub_data.index)
            sub_data = VENUS_Dataset(None, set_data=sub_data, device=self.device)
            sub_data.set_cols(self.cols_dict)
            datasets.append(sub_data)
            ratios = ratios[1:]
        return datasets


class Continuous_VENUS_Dataset(VENUS_Dataset):
    """
    A VENUS dataset used for fully convolutional neural network with 
        convolution as the time dimension. Each datapoint is a segment of 
        recorded features with possible different lengths. 
    
    @field displacement (int): The reduction of length in time dimension 
        due to convolution. 
    @field times (list[pandas.Interval]): A sequence of time intervals where 
        datapoint corresponds to. 
    @field segments (list[pandas.DataFrame]): The datapoint segments. 
    """
    def __init__(self, filename: str, device: torch.device = device, 
            csv_kwargs: dict = {}, drop_na: bool = False, 
            displacement: int = 0):
        """
        Constructor.
        @params filename (str): The file name where the data is located. If 
            it is an empty string, then the data is provided explicitly. 
            Supported file types include h5 and csv files. 
        @params device (torch.device): The device where the parameters are 
            stored. 
        @params drop_na (bool, DEFAULT False): True iff the NAN in the data 
            are dropped row-wise.
        @params csv_kwargs (dict): The argument provided to `pandas.read_csv` 
            when the data is loaded. 
        @params displacement (int, DEFAULT 0): The reduction of length in 
            time dimension due to convolution. 
            CONSTRAINT: displacement >= 0
        """
        super().__init__(filename, drop_na=drop_na, device=device)
        self.displacement = displacement
        self.times = None

    def set_times(self, times: Optional[list[pd.Interval]] = None) -> None:
        """
        Set up the each segment of data as datapoint. 
        @params times (Optional[list[pandas.Interval]]): A sequence of time 
            intervals where datapoint corresponds to. If not specified, 
            then the entire dataset is treated as a single datapoint.
        """
        if times is None:
            times = [pd.Interval(-np.inf, np.inf)]
        self.times = times
        self.segments = []
        for t in self.times:
            segment_idx = \
                self.data["unix_epoche_milliseconds"].map(t.__contains__)
            segment = self.data[segment_idx]
            segment = segment.sort_values(by=["unix_epoche_milliseconds"])
            segment = segment.reset_index(drop=True)
            self.segments.append(segment)

    def get_segment_lengths(self) -> list[int]:
        """
        Get the length of each segment. 
        @return (list[int]): The length of each segment. 
        """
        return [len(segment) for segment in self.segments]

    # def set_cols(self, cols_dict):
    #     self.cols_dict = cols_dict
    #     self.input_cols = input_cols
    #     self.output_cols = output_cols
    #     self.input = []
    #     self.output = []
    #     for segment in self.segments:
    #         input = segment[self.input_cols]
    #         output = segment[self.output_cols].iloc[self.displacement:]
    #         self.input.append(torch.tensor(
    #             input.to_numpy(), device=self.device, dtype=torch.float32
    #         ))
    #         self.output.append(torch.tensor(
    #             output.to_numpy(), device=self.device, dtype=torch.float32
    #         ))

    def set_displacement(self, displacement: int) -> None:
        """
        Set the displacement. 
        @params displacement (int): The reduction of length in time dimension 
            due to convolution. 
            CONSTRAINT: displacement >= 0
        """
        self.displacement = displacement

    def scale_output(self, scale):
        """
        Deprecated!
        """
        raise DeprecationWarning("The scale_output method is deprecated!")
        self.output = [out_seg * scale for out_seg in self.output]

    def __len__(self):
        """
        OVERRIDE
        """
        return len(self.segments)

    def __getitem__(self, i):
        """
        OVERRIDE
        @return (dict[str, torch.Tensor]): The datapoint with key `"input"` as 
            input features and `"output"` as output features. 
        """
        segment = self.segments[i]
        return {
            "input": torch.tensor(
                segment[self.input_cols].to_numpy(), 
                device=self.device, dtype=torch.float32
            ),
            "output": torch.tensor(
                segment[self.output_cols].iloc[self.displacement:].to_numpy(), 
                device=self.device, dtype=torch.float32
            )
        }

    def get_dataframes(self) -> dict[str, list[pd.DataFrame]]:
        """
        Get the dataframes for input and output features. 
        @return (dict[str, list[pd.DataFrame]]): input list of features and 
            output list of features. 
        """
        return {
            key:[segment[self.cols_dict[key]] for segment in self.segments] 
            for key in self.cols_dict.keys()
        }

    # def get_output_dataframe(self):
    #     return [segment[self.output_cols] for segment in self.segments]


class Stable_VENUS_h5_Dataset(torch.utils.data.Dataset):
    """
    Deprecated! Aggregate the dataset beforehand and use `VENUS_Dataset`. 
    """
    def __init__(self, file_name, agg_method=["mean"], device=device):
        raise DeprecationWarning("`Stable_VENUS_h5_Dataset` is deprecated!")
        super().__init__()
        self.data = pd.read_csv(file_name)
        self.data = self.data.groupby(by=["Sets"]).agg(agg_method)
        self.device = device

    def get_cols(self):
        return self.data.columns

    def set_cols(self, input_cols, output_cols):
        self.input_cols = input_cols
        self.input = self.data[self.input_cols]
        self.output_cols = output_cols
        self.output = self.data[self.output_cols]
        self.input = torch.tensor(
            self.input.values, device=self.device, dtype=torch.float32
        )
        self.output = torch.tensor(
            self.output.values, device=self.device, dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {
            "input": self.input[i],
            "output": self.output[i]
        }

    def get_input_dataframe(self):
        return self.data[self.input_cols]

    def get_output_dataframe(self):
        return self.data[self.output_cols]

def subset(dataset: torch.utils.data.Dataset, ratio: float, 
        seed: Optional[int] = None) -> torch.utils.data.Dataset:
    """
    Generate a random subset of a dataset. 
    @params dataset (torch.utils.data.Dataset): The original dataset.
    @params ratio (float): The ratio of the new dataset length over the 
        original dataset length. If ratio is more than 1, then no subsample 
        occurs.
    @params seed (Optional[int], DEFAULT None): The seed of the random 
        number generator. If not specified, then a random seed is used. 
    @return (torch.utils.data.Dataset): The subsampled dataset. 
    @error (ValueError): Error if the sampling ratio is no bigger than 0. 
    """
    if ratio <= 0:
        raise ValueError(
            f"Ratio of selection must be positive, but it is {ratio}")
    elif ratio < 1:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(dataset), int(len(dataset) * ratio), replace=False)
        dataset = torch.utils.data.Subset(dataset, idx)
    return dataset

def read_and_split_data_simple(val_proportion, trial=None):
    """
    Deprecated! Use filter and random split in `VENUS_Dataset`
    """
    raise DeprecationWarning("`read_and_split_data_simple` is deprecated!")
    data = pd.read_csv(CURRENT_DATA_FILE)
    if trial is not None:
        data = data[data["trial index"].isin(trial)]
    data = data.sample(frac=1)
    split_idx = int(val_proportion * len(data))
    return data[split_idx:], data[:split_idx]

def read_and_split_data_by_trial_simple(train_idx=None, val_idx=None):
    """
    Deprecated! Use filter in `VENUS_Dataset`
    """
    raise DeprecationWarning("`read_and_split_data_by_trial_simple` is "
        "deprecated!")
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
    val_idx = data["trial index"].isin(val_idx)
    return data[train_idx], data[val_idx]

def train_val_split(dataset: torch.utils.data.Dataset, val_ratio: float) \
    -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Randomly do a train-validation split.
    @params dataset (torch.utils.data.Dataset): The original dataset.
    @params val_ratio (float): The validation ratio. 
        CONSTRAINT: val_ratio >= 0 and val_ratio <= 1.
    @return (tuple[torch.utils.data.Subset, torch.utils.data.Subset]): A 
        list containg the training dataset and validation dataset in sequence. 
    @error (ValueError): Error if validation ratio is smaller than 0 or 
        greater than 1. 
    """
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError(f"Invalid validation split ratio: {val_ratio}")
    val_length = int(len(dataset) * val_ratio)
    train_length = len(dataset) - val_length
    return tuple(
        torch.utils.data.random_split(dataset, [train_length, val_length])
    )

def split_dataset_by_val_idx(
        dataset: torch.utils.data.Dataset, val_idx: list[int]
    ) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Do a train-validation split with given validation dataset indices. 
    @params dataset (torch.utils.data.Dataset): The original dataset.
    @params val_idx (list[int]): The validation dataset indices. 
        CONSTRAINT: all([i < len(dataset) for i in val_idx])
    @return (tuple[torch.utils.data.Subset, torch.utils.data.Subset]): A 
        list containg the training dataset and validation dataset in sequence. 
    @error (ValueError): Error if validation dataset indices are out of 
        bounds. 
    """
    if not all([i < len(dataset) for i in val_idx]):
        raise ValueError("Validation dataset indices out of bounds!")
    train_idx = [i for i in range(len(dataset)) if i not in val_idx]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset

# Old name conversion
Full_VENUS_h5_Dataset = VENUS_Dataset
Full_Continuous_VENUS_h5_Dataset = Continuous_VENUS_Dataset