import numpy as np
import os
import pandas as pd
import re
import shutil

MONITOR_HEADER = ["block time", "run time", "fcv1_i", "extraction_i", "bias_i",
    "inj_mbar", "ext_mbar", "inj_i", "ext_i", "mid_i", "sext_i",
    "x_ray_source"]
DATA_DIR = "VENUS_data"
DATA_CSV = "VENUS_data.csv"
FILE_DATA_CSV = "file_data.csv"


def organize_file_data_csv():
    file_data = pd.read_csv("old_file_data.csv")
    original_dir = "VENUS_data_2022.02.25-2022.02.27"

    def join_parent_dir(name):
        if pd.isna(name):
            return ""
        return os.path.join(original_dir, name)

    file_data["actual monitor file"] = file_data.apply(
        lambda d: join_parent_dir(d["actual monitor file"]), axis=1
    )
    file_data["actual std file"] = file_data.apply(
        lambda d: join_parent_dir(d["actual std file"]), axis=1
    )
    file_data.to_csv("old_file_data.csv")

def rename_monitor_time_step_col():
    file_data = pd.read_csv("old_file_data.csv")
    file_data = file_data.rename(columns={"time step": "monitor time step"})
    file_data.to_csv("old_file_data.csv")

def add_trial():
    file_data = pd.read_csv("old_file_data.csv")
    file_data["trial"] = 1
    columns = file_data.columns
    print(columns)
    columns = columns[[7]+list(range(2,7))]
    print(columns)
    file_data = file_data[columns]
    file_data.to_csv("old_file_data.csv", index=False)

def rename_cols():
    file_data = pd.read_csv("old_file_data.csv")
    file_data = file_data.rename(columns={
        "actual monitor file": "monitor file",
        "actual std file": "std file"
    })
    file_data.to_csv("old_file_data_test.csv", index=False)

def get_creator(fname):
    if "harvey" in fname:
        return "harvey"
    if "wenhan" in fname:
        return "wenhan"
    return ""

def data_mar18():
    original_dir = "VENUS_data_2022.03.18-2022.03.20"
    trial = 1

    files = [os.path.join(original_dir, f) for f in os.listdir(original_dir)
        if ("dump" in f) or ("monitor" in f)]
    monitor_files = [d for d in files if "monitor" in d]
    std_files = [d for d in files if "dump" in d]
    monitor_files = sorted(monitor_files, key = lambda d: int(d[-10:]))
    std_files = sorted(std_files, key = lambda d: int(d[-10:]))
    assert len(monitor_files) == len(std_files)
    data = []

    for i in range(len(monitor_files)):
        assert get_creator(monitor_files[i]) == get_creator(std_files[i])
        data_d = {
            "trial index": trial,
            "file index": i,
            "monitor file": monitor_files[i],
            "std file": std_files[i],
            "creator": get_creator(monitor_files[i]),
            "monitor time step": int(monitor_files[i][-10:]),
            "std time step": int(std_files[i][-10:])
        }
        data.append(data_d)
    data = pd.DataFrame(data)
    print(data.head(5))
    return data

def merge_data_mar18():
    data = pd.read_csv("old_file_data.csv")
    new_data = data_mar18()
    data = pd.concat([data, new_data])
    data.to_csv("old_file_data_test.csv", index=False)

def merge_data_apr():
    time_step = lambda f: int(f[-10:])
    original_dir = "VENUS_data_2022.04.29-2022.05.01"
    data = pd.read_csv("old_file_data.csv")
    data_fname = os.listdir(original_dir)
    std_files = [d for d in data_fname if "dump" in d]
    monitor_files = [d for d in data_fname if "monitor" in d]
    std_files = sorted(std_files, key=time_step)
    monitor_files = sorted(monitor_files, key=time_step)
    assert len(std_files) == len(monitor_files)
    std_creator = [get_creator(d) for d in std_files]
    monitor_creator = [get_creator(d) for d in monitor_files]
    assert std_creator == monitor_creator
    std_files = [os.path.join(original_dir, d) for d in std_files]
    monitor_files = [os.path.join(original_dir, d) for d in monitor_files]
    new_data = []

    for i in range(len(monitor_files)):
        data_d = {
            "trial index": 2,
            "file index": i,
            "monitor file": monitor_files[i],
            "std file": std_files[i],
            "creator": std_creator[i],
            "monitor time step": int(monitor_files[i][-10:]),
            "std time step": int(std_files[i][-10:])
        }
        new_data.append(data_d)
    new_data = pd.DataFrame(new_data)
    data = pd.concat([data, new_data])
    data.to_csv("old_file_data.csv", index=False)


if __name__ == "__main__":
    # merge_data_apr()
    pass
