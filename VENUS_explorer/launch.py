"""
Launch the main function with specified parameters. 
    Date: May 2023
    Author: Wenhan Sun
"""

from neural_net_explorer import *
from random_forest_analysis import *
import os

def launch_train_generalizability_full():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    data_dict["train_ids"] = []
    data_dict["val_ids"] = [6.5]
    # data_dict["not_input_cols"] = [
    #     "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
    #     "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    # ]
    data_dict["not_input_cols"] = None
    data_dict["pca_dim"] = None
    data_dict["intermediate_layer_sizes"] = [64, 16, 4]
    data_dict["num_epochs"] = 500
    data_dict["lr_schedular_kwargs"] = {"patience":3, "factor":0.9}
    data_dict["loss_fns"] = ["MSE", "MAE", "Rel_MSE", "Rel_MAE"]

    power_set = lambda s: itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    for cols in power_set(SECONDARY_COLS):
        data_dict["input_cols"] = PRIMARY_COLS + list(cols)
        data_dict["img_folder"] = f"gen_plots {cols}"
        train_generalizability(data_dict)

def launch_train_generalizability():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    
    data_dict["not_input_cols"] = [
        "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
        "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    ]
    data_dict["input_cols"] = None
    data_dict["pca_dim"] = None
    data_dict["intermediate_layer_sizes"] = [256, 64, 16]
    data_dict["num_epochs"] = 500
    # data_dict["num_epochs"] = 5
    data_dict["lr_schedular_kwargs"] = {"patience":3, "factor":0.9}
    data_dict["loss_fns"] = ["MSE", "MAE", "Rel_MSE", "Rel_MAE"]
    data_dict["quantile_filter"] = None
    data_dict["learning_rate"] = 0.5

    # data_dict["input_cols"] = PRIMARY_COLS + SECONDARY_COLS
    ids = [1, 2, 3, 5, 6, 6.5, 7, 7.5, 8, 9, 10]
    for i in range(len(ids)):
        data_dict["train_ids"] = ids[:i] + ids[i+1:]
        data_dict["val_ids"] = [ids[i]]
        data_dict["img_folder"] = os.path.join("full_gen_plots", f"gen_plots_{ids[i]}")
        data_dict["csv_file_val_data_w_prediction"] = ""
        train_generalizability(data_dict)

def launch_train_nn():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    data_dict["run_ids_name"] = "all"
    data_dict["val_frac"] = 0.2
    
    data_dict["pca_dim"] = None
    data_dict["intermediate_layer_sizes"] = [256, 64, 8]
    data_dict["num_epochs"] = 500
    # data_dict["num_epochs"] = 5
    data_dict["learning_rate"] = 0.5
    data_dict["lr_schedular_kwargs"] = {"patience":3, "factor":0.9}
    data_dict["loss_fns"] = ["MSE"]

    data_dict["input_cols"] = None

    data_dict["not_input_cols"] = [
        "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
        "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    ]
    data_dict["img_folder"] = f"all_data with_xray"
    data_dict["csv_file_val_data_w_prediction"] = ""
    train_nn(data_dict)
    # data_dict["not_input_cols"] = [
    #     "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
    #     "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time", 
    #     "x_ray_source", "x_ray_exit"
    # ]
    # data_dict["img_folder"] = f"all_data without_xray"
    # train_nn(data_dict)

def launch_trial_gen_filter():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    data_dict["train_ids"] = [5]
    data_dict["val_ids"] = [7, 7.5]
    # data_dict["not_input_cols"] = [
    #     "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
    #     "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    # ]
    data_dict["not_input_cols"] = None
    data_dict["pca_dim"] = None
    data_dict["intermediate_layer_sizes"] = [64, 16, 4]
    data_dict["num_epochs"] = 500
    data_dict["learning_rate"] = 0.5
    data_dict["lr_schedular_kwargs"] = {"patience":3, "factor":0.9}
    data_dict["loss_fns"] = ["MSE"]

    data_dict["quantile_filter"] = (0, 1)

    data_dict["input_cols"] = PRIMARY_COLS + SECONDARY_COLS
    data_dict["img_folder"] = f"gen_plots_filtered SECONDARY {data_dict['val_ids']} {data_dict['quantile_filter']}"
    train_generalizability(data_dict)

def launch_compute_generalizability():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    # data_dict["not_input_cols"] = [
    #     "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
    #     "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    # ]
    data_dict["not_input_cols"] = None
    data_dict["quantile_filters"] = [(0,1),(0.01,0.99),(0.05,0.95),(0.1,0.9)]
    data_dict["input_cols"] = PRIMARY_COLS
    data_dict["save_csv_file"] = os.path.join("Generalizability", "PRIMARY", "compute_generalizability.csv")
    compute_generalizability(data_dict)
    generalizability_plot(data_dict)
    data_dict["input_cols"] = PRIMARY_COLS + SECONDARY_COLS
    data_dict["save_csv_file"] = os.path.join("Generalizability", "SECONDARY", "compute_generalizability.csv")
    compute_generalizability(data_dict)
    generalizability_plot(data_dict)

def launch_train_gen_with_prediction():
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    data_dict["train_ids"] = [5]
    data_dict["val_ids"] = [7, 7.5]
    
    data_dict["pca_dim"] = None
    data_dict["intermediate_layer_sizes"] = [64, 16, 4]
    data_dict["num_epochs"] = 500
    data_dict["learning_rate"] = 0.5
    data_dict["lr_schedular_kwargs"] = {"patience":3, "factor":0.9}
    data_dict["loss_fns"] = ["MSE"]
    data_dict["not_input_cols"] = None
    data_dict["quantile_filter"] = None
    

    data_dict["input_cols"] = PRIMARY_COLS + SECONDARY_COLS
    data_dict["img_folder"] = f"SECONDARY_w_prediction"
    data_dict["csv_file_val_data_w_prediction"] = "SECONDARY_val_data_w_prediction.csv"
    train_generalizability(data_dict)

def launch_train_rf():
    """
    Train the extra-trees regressor. 
    """
    data_dict = {}
    data_dict["csv_file"] = CSV_DATA_FILE
    data_dict["csv_kwargs"] = {"header": [0, 1]}
    data_dict["run_ids_name"] = "all"
    data_dict["val_frac"] = 0.2

    data_dict["input_cols"] = None

    data_dict["not_input_cols"] = [
        "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
        "batman_i", "batman_field", "run_id", "start_time", "stop_time", "time"
    ]
    # data_dict["img_folder"] = f"all_data with_xray"
    train_rf(data_dict)

if __name__ == "__main__":
    # launch_train_generalizability()
    launch_train_nn()
    # launch_trial_gen_filter()
    # launch_compute_generalizability()
    # launch_train_gen_with_prediction()
    # launch_train_rf()