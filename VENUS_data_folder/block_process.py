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

class Monitor_File:
    def __init__(self, file_name):
        self.file_name = file_name
        self.empty = self.is_empty()
        if self.empty:
            self.separation_loc = None
            self.num_block = 0
        else:
            data = np.loadtxt(file_name, usecols=0)
            self.separation_loc = find_separation_loc(data)
            self.num_block = len(self.separation_loc) - 1

    def create_separate_files(self):
        if self.empty:
            return
        for i in range(self.num_block):
            data = self.get_block_data(i)
            block_file_name = self.file_name + f"_block_{i:02d}.csv"
            data = pd.DataFrame(data=data, columns=MONITOR_HEADER)
            data.to_csv(block_file_name, index=False)

    def is_empty(self):
        f = open(self.file_name, "r")
        return re.match(r"^\s*$", f.read())

    def get_block_data(self, idx):
        if idx < 0 or idx >= self.num_block:
            raise IndexError("Index error in Monitor_File")
        data = np.loadtxt(self.file_name)
        return data[self.separation_loc[idx]: self.separation_loc[idx+1]]

def find_separation_loc(data):
    data = np.diff(data)
    idx = np.where(data < 0)[0] + 1
    return np.concatenate(([0], idx, [data.shape[0] + 1]))

class Std_File:
    def __init__(self, file_name):
        self.file_name = file_name

    def create_separate_files(self):
        f = open(self.file_name, "r")
        content = re.split(r"\n\s*\n", f.read())
        content = list(filter(self.valid_content, content))
        for i, c in enumerate(content):
            f = open(self.file_name + f"_block_{i:02d}", "w")
            f.write(c)
            f.close()

    def valid_content(self, s0):
        return "in change" in s0 and "average current for 10 s:" in s0

class Std_Block_File:
    """
    A std_out block file.
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def get_content(self):
        return open(self.file_name, "r").read()


    def valid_content(self, s0):
        return "in change" in s0 and "average current for 10 s:" in s0

    def process(self):
        block = self.get_content()
        result = {}
        block = block.splitlines()
        inow = self.extract_i_init(block, "now")
        if inow is not None:
            result["init_inow_inj"] = inow[0]
            result["init_inow_ext"] = inow[1]
            result["init_inow_mid"] = inow[2]
        iaim = self.extract_i_init(block, "aim")
        if iaim is not None:
            result["init_iaim_inj"] = iaim[0]
            result["init_iaim_ext"] = iaim[1]
            result["init_iaim_mid"] = iaim[2]
        ioff = self.extract_i_init(block, "off")
        if ioff is not None:
            result["init_ioff_inj"] = ioff[0]
            result["init_ioff_ext"] = ioff[1]
            result["init_ioff_mid"] = ioff[2]

        inj_i = self.extract_i_final(block, "inj")
        if inj_i is not None:
            result["final_inow_inj"] = inj_i[0]
            result["final_igoal_inj"] = inj_i[1]
        ext_i = self.extract_i_final(block, "ext")
        if ext_i is not None:
            result["final_inow_ext"] = ext_i[0]
            result["final_igoal_ext"] = ext_i[1]
        mid_i = self.extract_i_final(block, "mid")
        if mid_i is not None:
            result["final_inow_mid"] = mid_i[0]
            result["final_igoal_mid"] = mid_i[1]

        result["beam current"] = self.extract_beam(block)
        result["time out"] = self.extract_timeout(block)

        set_time, monitor_time = self.extract_time(block)
        if set_time is not None:
            result["set time"] = set_time
        if monitor_time is not None:
            result["monitor time"] = monitor_time
        return result


    def extract_i_init(self, block, name):
        line = list(filter(lambda s0: s0[:4] == f"I{name}", block))
        # assert len(line) <= 1, f"Multiple initial I{name} line detected!"
        if len(line) == 0:
            return None
        line = line[0]
        vals = re.findall(
            r"\[\s*(\d+\.?\d*e?\-?\d*)\s+(\d+\.?\d*e?\-?\d*)\s+(\d+\.?\d*e?\-?\d*)\s*\]",
            line)[0]
        return tuple(map(float, vals))

    def extract_i_final(self, block, name):
        line = list(filter(lambda s0: s0[:11] == f"{name} to goal", block))
        if len(line) == 0:
            return None
        line = line[0]
        now_val = float(re.findall(r"Inow:\s+(\d+\.?\d*e?\-?\d*)", line)[0])
        goal_val = float(re.findall(r"Igoal:\s+(\d+\.?\d*e?\-?\d*)", line)[0])
        return now_val, goal_val

    def extract_beam(self, block):
        line = list(filter(lambda s0: "average current for 10 s:" in s0, block))
        assert len(line) == 1, f"Beam current specification not valid!"
        line = line[0]
        beam_current = float(re.findall(r":\s+(\d+\.?\d*e?\-?\d*)", line)[0])
        if beam_current < 1:
            beam_current *= 1e6
        return beam_current

    def extract_timeout(self, block):
        line = list(filter(lambda s0: "timed out" in s0, block))
        return line != []

    def extract_average_time(self, block):
        line = list(filter(lambda s0: "seconds to do averaging" in s0, block))
        if len(line) == 0:
            return None
        line = line[0]
        val = float(re.findall(r"\d+\.?\d*", line)[0])
        return val

    def extract_time(self, block):
        line = list(filter(lambda s0: "seconds to set superconductors" in s0, block))
        if len(line) == 0:
            return None, None
        set_time = float(re.findall(r"\d+\.?\d*", line[0])[0])
        if len(line) >= 2:
            monitor_time = float(re.findall(r"\d+\.?\d*", line[1])[0])
        else:
            monitor_time = None
        return set_time, monitor_time

def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def organize_files(old_file_data_name):
    """
    Requires old_file_data.csv
    """
    file_data = pd.read_csv(old_file_data_name)
    mkdir(DATA_DIR)
    data_lst = []
    for data in file_data.iloc:
        std_f = data["std file"]
        monitor_f = data["monitor file"]
        file_idx = data["file index"]
        trial_idx = data["trial index"]
        run_dir = os.path.join(DATA_DIR,
            f"trial_{trial_idx:02d}_run_{file_idx:02d}")
        mkdir(run_dir)
        new_std_f = ""
        new_monitor_f = ""
        if type(std_f) is not float:
            new_std_f = os.path.join(run_dir, "std_file")
            shutil.copyfile(std_f, new_std_f)
        if type(monitor_f) is not float:
            new_monitor_f = os.path.join(run_dir, "monitor_file")
            shutil.copyfile(monitor_f, new_monitor_f)
        new_data = {
            "trial index": trial_idx,
            "file index": file_idx,
            "monitor file": new_monitor_f,
            "std file": new_std_f,
            "creator": data["creator"],
            "monitor time step": data["monitor time step"],
            "std time step": data["std time step"]
        }
        data_lst.append(new_data)
    data = pd.DataFrame(data_lst)
    data.to_csv(FILE_DATA_CSV, index=False)

def construct_block_files():
    file_data = pd.read_csv(FILE_DATA_CSV)
    for d in file_data.iloc:
        std_f = d["std file"]
        monitor_f = d["monitor file"]
        if not pd.isna(std_f):
            std_file = Std_File(std_f)
            std_file.create_separate_files()
        if not pd.isna(monitor_f):
            monitor_file = Monitor_File(monitor_f)
            monitor_file.create_separate_files()

def construct_main_csv():
    """
    Construct VENUS_data.csv from file_data.csv by extracting data.
    """
    file_data = pd.read_csv(FILE_DATA_CSV)
    result = []

    for file_d in file_data.iloc:
        std_f = file_d["std file"]
        monitor_f = file_d["monitor file"]
        file_idx = file_d["file index"]
        trial_idx = file_d["trial index"]
        dir_name = os.path.join(DATA_DIR,
            f"trial_{trial_idx:02d}_run_{file_idx:02d}")
        if pd.isna(std_f) or pd.isna(monitor_f):
            continue
        dir_files = os.listdir(dir_name)
        std_block_files = sorted([os.path.join(dir_name, n) for n in dir_files
            if "std_file_block" in n])
        monitor_block_files = sorted([os.path.join(dir_name, n)
            for n in dir_files if "monitor_file_block" in n and ".csv" in n])
        assert len(std_block_files) == len(monitor_block_files), \
            f"Number of blocks not the same for {dir_name}!"
        num_block = len(std_block_files)
        for idx_b in range(num_block):
            std_block_f = os.path.join(dir_name,
                f"std_file_block_{idx_b:02d}")
            monitor_block_f = os.path.join(dir_name,
                f"monitor_file_block_{idx_b:02d}.csv")
            std_block_file = Std_Block_File(std_block_f)
            data = {
                "trial index": trial_idx,
                "file index": file_idx,
                "block index": idx_b,
                "creator": file_d["creator"],
                "monitor block path": monitor_block_f,
                "std block path": std_block_f
            }
            data.update(std_block_file.process())
            result.append(data)
    result = pd.DataFrame(result)
    result.to_csv(DATA_CSV, index=False)

def main():
    organize_files("old_file_data.csv")
    construct_block_files()
    construct_main_csv()

if __name__ == "__main__":
    main()
