import pandas as pd
import matplotlib.pyplot as plt
from extract_current import extract_current_data

MONITOR_HEADER = ["block time", "run time", "fcv1_i", "extraction_i", "bias_i",
    "inj_mbar", "ext_mbar", "inj_i", "ext_i", "mid_i", "sext_i",
    "x_ray_source"]
DATA_DIR = "VENUS_data"
DATA_CSV = "VENUS_data.csv"
FILE_DATA_CSV = "file_data.csv"

def load_monitor_data(data, i, cols):
    return pd.read_csv(data["monitor block path"][i])[cols]

def cov_data(x, y):
    x_avg = x.mean()
    y_avg = y.mean()
    return np.mean((x-x_avg) * (y-y_avg))

def covariance_field():
    data = pd.read_csv(DATA_CSV)
    data = extract_current_data(data)
    num_data = len(data)
    cov = np.zeros((num_data, num_data))
    for i in range(num_data):
        data_i = load_monitor_data(data, i, "fcv1_i")
        for j in range(i, num_data):
            data_j = load_monitor_data(data, j, "fcv1_i")
            cov[i,j] = np.mean((x-x_avg) * (y-y_avg))
            cov[j,i] = cov[i,j]
    return data[["inj_avg", "ext_avg", "mid_avg"]], cov



def plot_std():
    data = pd.read_csv("VENUS_current_data.csv")
    data = data["beam_std"]
    plt.hist(data, bins="sqrt")
    plt.show()

if __name__ == "__main__":
    covariance_field()
