import pandas as pd
import numpy as np

CURRENT_NAMES = ["inj", "ext", "mid"]

def close_current(i1, i2, thres = 0.01):
    return np.abs(i1 - i2) <= thres * np.abs(i1 + i2) / 2

def current_converge(data, name):
    final_now_name = f"final_inow_{name}"
    final_goal_name = f"final_igoal_{name}"
    time_out = data["time out"]
    f_inow = data[final_now_name]
    f_igoal = data[final_goal_name]
    if pd.isna(f_inow):
        assert pd.isna(f_igoal)
        return "i"
    elif close_current(f_inow, f_igoal):
            return "f"
    else:
        return "n"

def current_avg(data, name):
    if name in CURRENT_NAMES:
        flag = data[f"{name}_flag"]
        if flag == "f":
            return data[f"final_inow_{name}"]
        elif flag == "i":
            return data[f"init_inow_{name}"]
        else:
            return pd.NA


def get_monitor_avg(data, name):
    return pd.read_csv(data["monitor block path"])[name].mean()

def get_monitor_std(data, name):
    return pd.read_csv(data["monitor block path"])[name].std()

def current_std(data, name):
    return get_monitor_std(data, f"{name}_i")


def extract_current_data(data):
    for name in CURRENT_NAMES:
        # data[f"{name}_flag"] = data.apply(
        #     lambda x: current_converge(x, name), axis=1
        # )
        data[f"{name}_avg"] = data.apply(
            lambda x: get_monitor_avg(x, f"{name}_i"), axis=1
        )
        data[f"{name}_std"] = data.apply(
            lambda x: get_monitor_std(x, f"{name}_i"), axis=1
        )
    data["bias_avg"] = data.apply(
        lambda x: get_monitor_avg(x, "bias_i"), axis=1
    )
    data["bias_std"] = data.apply(
        lambda x: current_std(x, "bias"), axis=1
    )
    data["inj_p_avg"] = data.apply(
        lambda x: get_monitor_avg(x, "inj_mbar"), axis=1
    )
    data["inj_p_std"] = data.apply(
        lambda x: get_monitor_std(x, "inj_mbar"), axis=1
    )
    data["ext_p_avg"] = data.apply(
        lambda x: get_monitor_avg(x, "ext_mbar"), axis=1
    )
    data["ext_p_std"] = data.apply(
        lambda x: get_monitor_std(x, "ext_mbar"), axis=1
    )
    data["beam_avg"] = data.apply(
        lambda x: get_monitor_avg(x, "fcv1_i"), axis=1
    )
    data["beam_std"] = data.apply(
        lambda x: get_monitor_std(x, "fcv1_i"), axis=1
    )
    return data

def extract_valid_data(data):
    mask = ~data["time out"]
    # for name in CURRENT_NAMES:
    #     mask = mask & (data[f"{name}_flag"] != "n")
    return data[mask]

def get_data():
    return pd.read_csv("VENUS_data.csv")

def create_current_csv():
    data = get_data()
    data = extract_valid_data(extract_current_data(data))
    cols = ["trial index", "file index", "block index"]
    cols += ["beam_avg", "beam_std"]
    for name in CURRENT_NAMES:
        cols.extend([f"{name}_avg", f"{name}_std"])
    cols += ["bias_avg", "bias_std"]
    cols += ["inj_p_avg", "inj_p_std", "ext_p_avg", "ext_p_std"]
    # data = data[cols].rename(columns={"beam current": "beam_avg"})
    data = data[cols]
    data.to_csv("VENUS_current_data.csv", index=False)

if __name__ == "__main__":
    create_current_csv()
