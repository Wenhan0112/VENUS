"""
Test neural network package
    Date: May 2023
    Author: Wenhan Sun
"""

from VENUS_nn import VENUS_NN
import VENUS_dataset
import os

def test_venus_nn():
    STABLE_DATA_FILE = os.path.join(
        "..",
        "VENUS_data_folder",
        "VENUS_stable_data.csv"
    )
    csv_file = STABLE_DATA_FILE
    csv_kwargs = {"header": [0, 1]}
    dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    run_ids_name = "all"
    if run_ids_name == "all":
        run_ids = [1,2,3,5,6,7,7.5,8]
    elif run_ids_name == "first half":
        run_ids = [1,2,3]
    elif run_ids_name == "second half":
        run_ids = [5,6,7,7.5,8]
    else:
        run_ids = [float(run_ids_name)]
    dataset.filter_value(("run_id", ""), run_ids, mode="in")
    dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
    # print(run_ids_name, "Length of dataset", len(dataset))
    # return 

    cols_dict = {}
    not_input_cols = ["fcv1_i", "fcv1_in", "unix_epoche_milliseconds", "m_over_q", "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"]
    cols_dict["input"] = [
        col for col in dataset.get_cols() if col[0] not in not_input_cols and col[1] != "std"
    ]
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]
    data_point = {c:dataset.data[c][[5,7]] for c in cols_dict["input"]}

    # Construct wrapper
    venus_nn = VENUS_NN("model")
    # print(data_point)
    venus_nn.get_cols(1)
    print(venus_nn.predict(data_point))

def test_sole_venus_nn():
    venus_nn = VENUS_NN("model_all")
    # print(data_point)
    cols_dict = venus_nn.get_cols(0)
    data_point = {c:[1., 5.] for c in cols_dict["input"]}
    print(venus_nn.predict(data_point))

if __name__ == "__main__":
    test_sole_venus_nn()