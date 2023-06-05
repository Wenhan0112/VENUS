"""
Analysis of the neural networks on the dataset. 
    Date: May 2023
    Author: Wenhan Sun
"""

import h5py
import torch
from torch import nn
import torch.optim
import tqdm
import time
import re
import numpy as np
import os
import shutil
import itertools
import sklearn
import time
import datetime
import interval
import matplotlib.pyplot as plt
import pandas as pd
import physical_model
import trainer
import VENUS_dataset
import itertools
import matplotlib

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

"""
h5 file containing the VENUS data.  
Use csv file instead of h5 file. 
"""
H5_DATA_FILE = os.path.join(
    "..",
    "VENUS_data_folder",
    "VENUS_data.h5"
)

"""
csv file containing the VENUS data.  
Use this csv file instead of h5 file. 
"""
CSV_DATA_FILE = os.path.join(
    "..",
    "VENUS_data_folder",
    "VENUS_data.csv"
)

"""
The primarily important columns
"""
PRIMARY_COLS = [
    "inj_mbar",
    # "ext_mbar",
    "bias_v",
    "bias_i",
    # "k18_fw",
    # "k18_ref",
    # "g28_fw",
    # "puller_i",
    "extraction_i"
]

"""
The secondarily important columns
"""
SECONDARY_COLS = [
    "inj_i",
    "ext_i",
    "mid_i",
    "sext_i"
    # "x_ray_source",
    # "x_ray_exit"
]

def print_list(l: list) -> None:
    """
    Print each element in a list on a different line. 
    @params l (list): The input list. 
    """
    for i in l:
        print(i)
    print()

def check_run_id():
    """
    Check the ID of the weekends.
    # Note by Wenhan: This method current only check up to week 8. Modify
        `all_ids` to the current weekend IDs. 
    """
    lengths = []
    all_ids = [1,2,3,4,5,6,7,7.5,8]
    for run_id in all_ids:
        csv_file = CSV_DATA_FILE
        csv_kwargs = {"header": [0, 1]}
        dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
        # [print(x) for x in dataset.get_cols()]
        dataset.filter_value(("run_id", ""), [run_id])
        print("Length of dataset", run_id, len(dataset))
        lengths.append(len(dataset))
    csv_file = CSV_DATA_FILE
    csv_kwargs = {"header": [0, 1]}
    dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    dataset.filter_value(("run_id", ""), all_ids)
    # [print(x) for x in dataset.get_cols()]
    print("Length of dataset", len(dataset))
    print("Sum of subs datasets", sum(lengths))

def train_nn(data_dict):
    """
    Train the neural network. 
    """
    # Set up the VENUS dataset. 
    csv_file = data_dict["csv_file"]
    csv_kwargs = data_dict["csv_kwargs"]
    dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    # [print(x) for x in dataset.get_cols()]
    run_ids_name = data_dict["run_ids_name"]
    if run_ids_name == "all":
        run_ids = [1,2,3,5,6,6.5,7,7.5,8]
    elif run_ids_name == "first half":
        run_ids = [1,2,3]
    elif run_ids_name == "second half":
        run_ids = [5,6,6.5,7,7.5,8]
    else:
        run_ids = [float(run_ids_name)]
    dataset.filter_value(("run_id", ""), run_ids, mode="in")
    dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
    # print(run_ids_name, "Length of dataset", len(dataset))

    # Set up the column dictionary. 
    cols_dict = {}
    if data_dict["not_input_cols"]:
        not_input_cols = data_dict["not_input_cols"]
        cols_dict["input"] = [
            col for col in dataset.get_cols() if col[0] not in not_input_cols and col[1] != "std"
        ]
    elif data_dict["input_cols"]:
        input_cols = data_dict["input_cols"]
        input_cols = data_dict["input_cols"]
        cols_dict["input"] = [
            col for col in dataset.get_cols() if col[0] in input_cols and col[1] != "std"
        ]
        assert len(cols_dict["input"]) == len(input_cols)
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]
    # print_list(cols_dict["input"])
    dataset.set_cols(cols_dict)

    # Set up accesories. 
    val_frac = data_dict["val_frac"]
    pca_dim = data_dict["pca_dim"]
    input_dim = pca_dim or len(cols_dict["input"])
    
    layer_sizes = [input_dim] + data_dict["intermediate_layer_sizes"] + [len(cols_dict["output"])]
    print("Layer sizes:", layer_sizes)
    model = physical_model.Naive_Net(layer_sizes)
    truth_processor = physical_model.Scaler(1e6)

    # Construct the dataset and model. 
    dataset.construct_dataset()
    train_dataset, val_dataset = VENUS_dataset.train_val_split(dataset, val_frac)
    lr_scheduler_setter = lambda optimizer: \
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.9
        )
    loss_fns = data_dict["loss_fns"]
    model_trainer = trainer.Trainer(train_dataset, model, loss_fns,
        truth_processor=truth_processor, 
        lr_scheduler_setter=lr_scheduler_setter)
    model_trainer.set_validation(val_dataset, -1)
    num_epochs = data_dict["num_epochs"]

    # Train the model. 
    model_trainer.train(num_epochs, data_dict["learning_rate"], -1, verbose=0)
    # model_trainer.visualize_structure()

    img_folder = data_dict["img_folder"]
    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)
    os.makedirs(img_folder)

    # Plot loss curves
    for i, loss_fn in enumerate(loss_fns):
        plt.plot(model_trainer.val_losses[i])
        plt.yscale("log")
        plt.ylabel(f"Validation Losses ({loss_fn})")
        plt.xlabel("Epoch")
        plt.title(f"Validation Loss Curve ({loss_fn})")
        plt.tick_params(direction="in")
        plt.savefig(
            os.path.join(img_folder, f"stable_val_loss_{loss_fn}.png")
        )
        plt.show()
        plt.close()
    
    # Save loss values
    val_losses = np.array(model_trainer.val_losses).T[1:]
    train_losses = np.array(model_trainer.train_losses).T
    val_losses = pd.DataFrame(val_losses, columns=[f"val {loss_fn}" for loss_fn in loss_fns])
    train_losses = pd.DataFrame(train_losses, columns=[f"train {loss_fn}" for loss_fn in loss_fns])
    epoch = pd.DataFrame(np.arange(1, num_epochs+1), columns=["epoch"])
    losses = pd.concat([epoch, val_losses, train_losses], axis=1)
    losses.to_csv(os.path.join(img_folder, "stable_losses.csv"), index=False)
    
    # Visualize error
    for err_type in ["abs", "rel"]:
        savefig = os.path.join(img_folder, f"stable_{err_type}err_truth.png")
        model_trainer.err_plot_truth(["train", "val"], err_type=err_type, 
            savefig=savefig)
        savefig = os.path.join(img_folder, f"stable_{err_type}err_time.png")
        model_trainer.err_plot_time(["train", "val"], err_type=err_type, 
            savefig=savefig)
                


    # val_data_sub = next(iter(torch.utils.data.DataLoader(val_dataset, 20, shuffle=True)))
    # pred = trainer.predict(val_data_sub["input"]).detach().flatten().cpu().numpy()
    # truth = val_data_sub["output"].detach().flatten().cpu().numpy()
    # plt.plot(pred, "b.", label="Pred")
    # plt.plot(truth, "r.", label="Truth")
    # plt.xlabel("Index")
    # plt.ylabel("Beam current ($\mu$A)")
    # plt.tick_params(direction="in")
    # plt.legend()
    # plt.savefig("stable_val_sample.png")
    # plt.show()
    # plt.close()

    # trainer.err_histogram("train", savefig="train_stable_train_err_hist.png")
    # trainer.err_histogram("validation", savefig="train_stable_val_err_hist.png")


    # train_data = next(iter(torch.utils.data.DataLoader(train_dataset, len(train_dataset), shuffle=True)))
    # pred = trainer.predict(train_data["input"]).detach().flatten().cpu().numpy()
    # input = train_data["input"].detach().cpu().numpy()
    # output = train_data["output"].detach().cpu().numpy()
    # train_data = dataset.recreate_dataframe(input, output, pd.DataFrame(pred, columns=["Prediction"]))
    # train_data.to_csv("train_data_w_prediction.csv", index=False)

    # Save validation data with prediction. 
    if data_dict["csv_file_val_data_w_prediction"]:
        val_data = next(iter(torch.utils.data.DataLoader(
            val_dataset, len(val_dataset), shuffle=True)))
        pred = trainer.predict(val_data["input"])
        pred = pred.detach().flatten().cpu().numpy()
        input = val_data["input"].detach().cpu().numpy()
        output = val_data["output"].detach().cpu().numpy()
        val_data = dataset.recreate_dataframe(
            input, output, pd.DataFrame(pred, columns=["Prediction"])
        )
        val_data.to_csv(
            data_dict["csv_file_val_data_w_prediction"], index=False
        )
    
def train_all_combinations():
    """
    Deprecated! Use different launch method to train all combinations of 
        datasets.
    """
    raise DeprecationWarning("`train_all_combinations` is deprecated! Use " 
        "launch methods to train all combinations of datasets. ")
    ids = ["all", "first half", "second half"]
    ids += [str(i) for i in [1,2,3,5,6,7,7.5,8]]
    for run_id in ids:
        train_nn(run_id)

def train_generalizability(data_dict):
    """
    Train the neural network with training and validation data from different 
        weekend IDs. 
    """
    csv_file = data_dict["csv_file"]
    csv_kwargs = data_dict["csv_kwargs"]
    train_dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    # print(train_dataset.get_cols())
    train_ids = data_dict["train_ids"]

    train_dataset.filter_value(("run_id", ""), train_ids, mode="in")
    train_dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
    # train_dataset.filter_value(("unix_epoch_milliseconds", "mean"), "median", mode="<")

    val_dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    val_ids = data_dict["val_ids"]
    val_dataset.filter_value(("run_id", ""), val_ids, mode="in")
    val_dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
    # val_dataset.filter_value(("unix_epoch_milliseconds", "mean"), "median", mode=">")

    # print(run_ids_name, "Length of dataset", len(dataset))

    cols_dict = {}
    if data_dict["not_input_cols"]:
        not_input_cols = data_dict["not_input_cols"]
        cols_dict["input"] = [
            col for col in train_dataset.get_cols() if col[0] not in not_input_cols and col[1] != "std"
        ]
    elif data_dict["input_cols"]:
        input_cols = data_dict["input_cols"]
        cols_dict["input"] = [
            col for col in train_dataset.get_cols() if col[0] in input_cols and col[1] != "std"
        ]
        assert len(cols_dict["input"]) == len(input_cols)
    else:
        raise ValueError
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]
    # print_list(cols_dict["input"])
    train_dataset.set_cols(cols_dict)
    val_dataset.set_cols(cols_dict)

    print("Original train dataset length:", len(train_dataset))
    print("Original val dataset length:", len(val_dataset))
    if data_dict["quantile_filter"]:
        lower_quantile, upper_quantile = data_dict["quantile_filter"]
        lower_bound = train_dataset.get_statistics(cols_dict["input"], "quantile", lower_quantile)
        upper_bound = train_dataset.get_statistics(cols_dict["input"], "quantile", upper_quantile)
        val_dataset.filter_value(cols_dict["input"], lower_bound, mode=">=")
        val_dataset.filter_value(cols_dict["input"], upper_bound, mode="<=")
        print("Val dataset length:", len(val_dataset))

    pca_dim = data_dict["pca_dim"]
    input_dim = pca_dim or len(cols_dict["input"])
    
    layer_sizes = [input_dim] + data_dict["intermediate_layer_sizes"] + [len(cols_dict["output"])]
    print("Layer sizes:", layer_sizes)
    model = physical_model.Naive_Net(layer_sizes)
    truth_processor = physical_model.Scaler(1e6)

    train_dataset.construct_dataset()
    val_dataset.construct_dataset()
    loss_fns = data_dict["loss_fns"]
    lr_scheduler_setter = lambda optimizer: \
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.9
        )
    model_trainer = trainer.Trainer(train_dataset, model, loss_fns,
        truth_processor=truth_processor, 
        lr_scheduler_setter=lr_scheduler_setter)
    model_trainer.set_validation(val_dataset, -1)
    num_epochs = data_dict["num_epochs"]
    model_trainer.train(num_epochs, data_dict["learning_rate"], -1, verbose=0)
    # model_trainer.visualize_structure()

    # img_folder = f"Generalizability Plots (Id {train_ids}, {val_ids})"
    # img_folder = f"Generalizability Plots (Id {train_ids}, {val_ids}) {len(input_cols)}"
    img_folder = data_dict["img_folder"]
    if os.path.exists(img_folder):
        shutil.rmtree(img_folder)
    os.makedirs(img_folder)

    # Plot loss curves
    for i, loss_fn in enumerate(loss_fns):
        plt.plot(model_trainer.train_losses[i], label="train")
        plt.plot(model_trainer.val_losses[i], label="val")
        plt.yscale("log")
        plt.ylabel(f"Losses ({loss_fn})")
        plt.xlabel("Epoch")
        plt.title(f"Loss Curves ({loss_fn})")
        plt.tick_params(direction="in")
        plt.legend()
        plt.savefig(
            os.path.join(img_folder, f"stable_loss_{loss_fn}.png")
        )
        plt.show()
        plt.close()
    
    # Save loss values
    val_losses = np.array(model_trainer.val_losses).T[1:]
    train_losses = np.array(model_trainer.train_losses).T
    val_losses = pd.DataFrame(val_losses, columns=[f"val {loss_fn}" for loss_fn in loss_fns])
    train_losses = pd.DataFrame(train_losses, columns=[f"train {loss_fn}" for loss_fn in loss_fns])
    epoch = pd.DataFrame(np.arange(1, num_epochs+1), columns=["epoch"])
    losses = pd.concat([epoch, val_losses, train_losses], axis=1)
    losses.to_csv(os.path.join(img_folder, "stable_losses.csv"), index=False)
    
    # Visualize error
    for err_type in ["abs", "rel"]:
        savefig = os.path.join(img_folder, f"stable_{err_type}err_truth.png")
        model_trainer.err_plot_truth(["train", "val"], err_type=err_type, 
            savefig=savefig)
        savefig = os.path.join(img_folder, f"stable_{err_type}err_time.png")
        model_trainer.err_plot_time(["train", "val"], err_type=err_type, 
            savefig=savefig)
    
    if data_dict["csv_file_val_data_w_prediction"]:
        val_data = next(iter(torch.utils.data.DataLoader(val_dataset, len(val_dataset), shuffle=True)))
        pred = model_trainer.predict(val_data["input"]).detach().flatten().cpu().numpy()
        input = val_data["input"].detach().cpu().numpy()
        output = val_data["output"].detach().cpu().numpy()
        time = val_data["time"].detach().cpu().numpy()
        val_data = val_dataset.recreate_dataframe({"input":input, "output":output, "time":time}, pd.DataFrame(pred, columns=["Prediction"]))
        val_data.to_csv(data_dict["csv_file_val_data_w_prediction"], index=False)

def save_trainer(run_ids_name="all"):
    """
    Save the trainer after training. 
    """
    csv_file = CSV_DATA_FILE
    csv_kwargs = {"header": [0, 1]}
    dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
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
    not_input_cols = ["fcv1_i", "fcv1_in", "unix_epoch_milliseconds", 
        "m_over_q", "batman_i", "batman_field", "run_id", "start_time", 
        "stop_time", "time"]
    cols_dict["input"] = [
        col for col in dataset.get_cols() 
        if col[0] not in not_input_cols and col[1] != "std"
    ]
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]
    dataset.set_cols(cols_dict)

    val_frac = 0.2
    pca_dim = None
    input_dim = pca_dim or len(cols_dict["input"])
    
    layer_sizes = [input_dim, 256, 64, 8, len(cols_dict["output"])]
    # print("Layer sizes:", layer_sizes)
    model = physical_model.Naive_Net(layer_sizes)
    truth_processor = physical_model.Scaler(1e6)

    dataset.construct_dataset()
    train_dataset, val_dataset = \
        VENUS_dataset.train_val_split(dataset, val_frac)
    loss_fns = ["MSE"]
    lr_scheduler_setter = lambda optimizer: \
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.9
        )
    model_trainer = trainer.Trainer(train_dataset, model, loss_fns,
        truth_processor=truth_processor, 
        lr_scheduler_setter=lr_scheduler_setter)
    model_trainer.set_validation(val_dataset, -1)
    
    num_epochs = 500
    model_trainer.train(num_epochs, 0.5, -1, verbose=0)
    model_trainer.save_model(f"model_{run_ids_name}", cols_dict=cols_dict)

def bar_plot_errors():
    """
    Deprecated!
    """
    raise DeprecationWarning()
    def get_loss_fns(keys):
        val_keys = [k for k in keys if "val " in k]
        train_keys = [k for k in keys if "train " in k]
        val_loss_fns = [k[4:] for k in val_keys]
        train_loss_fns = [k[6:] for k in train_keys]
        assert set(val_loss_fns) == set(train_loss_fns)
        return val_loss_fns


    main_folder = os.path.join("..", "VENUS_data_folder", "stable_plots")
    id_folders = sorted(os.listdir(main_folder))
    ids = [re.findall(r"\(Id ([\w\s\.]+)\)", f)[0] for f in id_folders]
    id_folders = [os.path.join(main_folder, f) for f in id_folders]
    losses_files = [os.path.join(f, "stable_losses.csv") for f in id_folders]
    losses = [pd.read_csv(f) for f in losses_files]
    epochs = [l["epoch"].max() for l in losses]
    losses = [l[l["epoch"] == l["epoch"].max()].drop(columns="epoch") 
        for l in losses]
    [l.insert(0, "id", id) for l, id in zip(losses, ids)]
    losses = pd.concat(losses, ignore_index=True)
    # print(losses)
    loss_fns = get_loss_fns(losses.columns)
    for loss_fn in loss_fns:
        train_data = losses[f"train {loss_fn}"]
        val_data = losses[f"val {loss_fn}"]
        width = 0.3
        train_index = np.arange(len(train_data))
        val_index = train_index + width
        fig, ax = plt.subplots()
        train_bars = ax.bar(train_index, 
            train_data, width=width, label="Train")
        ax.bar_label(train_bars, 
            [str(v)[:6] for v in train_data], rotation=90, padding=1)
        val_bars = ax.bar(val_index, val_data, width=width, label="Val")
        ax.bar_label(val_bars, 
            [str(v)[:6] for v in val_data], rotation=90, padding=1)
        ax.margins(y=0.2)
        ax.set_yscale("log")
        ax.set_xticks((train_index + val_index) / 2, ids, rotation=45)
        ax.set_ylabel(loss_fn)
        ax.legend()
        fig.savefig(f"error_bars_{loss_fn}", bbox_inches="tight")
        fig.show()

def params_generalizability():
    """
    Plot the parameters as a function of time. 
    """
    csv_file = CSV_DATA_FILE
    csv_kwargs = {"header": [0, 1]}
    dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
    ids = [5, 7, 8]
    dataset.filter_value(("run_id", ""), ids, mode="in")

    cols_dict = {}
    not_input_cols = [
        "fcv1_i", "fcv1_in", "unix_epoch_milliseconds", "m_over_q", 
        "batman_i", "batman_field", "run_id", "start_time", "stop_time", 
        "time"
    ]
    cols_dict["input"] = [
        col for col in dataset.get_cols() 
        if col[0] not in not_input_cols and col[1] != "std"
    ]
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]

    dataset.set_cols(cols_dict)
    inputs = dataset.get_dataframe("input")
    outputs = dataset.get_dataframe("output")
    time = dataset.get_dataframe("time")[cols_dict["time"][0]]
    img_dir = "input_generalizability"
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)

    for k in cols_dict["input"]:
        fig, ax = plt.subplots()
        ax.scatter(time, inputs[k], s=1)
        ax.set_xlabel("Time")
        ax.set_ylabel(k)
        ax.tick_params(direction="in")
        fig.savefig(os.path.join(img_dir, k[0]+" "+k[1]))
        fig.show()
        plt.close()
    fig, ax = plt.subplots()
    ax.scatter(time, outputs[cols_dict["output"][0]], s=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Beam current")
    ax.tick_params(direction="in")
    fig.savefig(os.path.join(img_dir, "Beam current"))
    fig.show()
    plt.close()

def compute_generalizability(data_dict):
    """
    Compute the generalizability of parameters between different weekend IDs. 
    """
    csv_file = data_dict["csv_file"]
    csv_kwargs = data_dict["csv_kwargs"]
    ids = [1,2,3,5,6,6.5,7,7.5,8,9,10]

    data = []
    
    for train_id in ids:
        train_dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
        train_dataset.filter_value(("run_id", ""), train_id, mode="=")
        train_dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
        train_dataset_len = len(train_dataset)

        cols_dict = {}
        if data_dict["not_input_cols"]:
            not_input_cols = data_dict["not_input_cols"]
            cols_dict["input"] = [
                col for col in train_dataset.get_cols() if col[0] not in not_input_cols and col[1] != "std"
            ]
        elif data_dict["input_cols"]:
            input_cols = data_dict["input_cols"]
            cols_dict["input"] = [
                col for col in train_dataset.get_cols() if col[0] in input_cols and col[1] != "std"
            ]
            assert len(cols_dict["input"]) == len(input_cols)
        else:
            raise ValueError

        bounds = []
        prev_quantile = (0, 1)
        for lower_quantile, upper_quantile in data_dict["quantile_filters"]:
            assert prev_quantile[0] <= lower_quantile and prev_quantile[1] >= upper_quantile
            prev_quantile = (lower_quantile, upper_quantile)
            lower_bound = train_dataset.get_statistics(cols_dict["input"], "quantile", lower_quantile)
            upper_bound = train_dataset.get_statistics(cols_dict["input"], "quantile", upper_quantile)
            bounds.append((lower_bound, upper_bound))
        for val_id in ids:
            row_data = {"Train ID": train_id, "Val ID": val_id, "Train length": train_dataset_len}
            
            # train_dataset.filter_value(("unix_epoch_milliseconds", "mean"), "median", mode="<")

            val_dataset = VENUS_dataset.VENUS_Dataset(csv_file, csv_kwargs=csv_kwargs)
            val_dataset.filter_value(("run_id", ""), val_id, mode="=")
            val_dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
            # val_dataset.filter_value(("unix_epoch_milliseconds", "mean"), "median", mode=">")

            # print(run_ids_name, "Length of dataset", len(dataset))
            # return 

            row_data["Val length"] = len(val_dataset)
            for i in range(len(bounds)):
                lower_quantile, upper_quantile = data_dict["quantile_filters"][i]
                val_dataset.filter_value(cols_dict["input"], bounds[i][0], mode=">=")
                val_dataset.filter_value(cols_dict["input"], bounds[i][1], mode="<=")
                row_data[f"Quantile ({lower_quantile},{upper_quantile})"] = len(val_dataset)
            data.append(row_data)
    data = pd.DataFrame(data)
    csv_dir = os.path.dirname(data_dict["save_csv_file"])
    if os.path.exists(csv_dir):
        shutil.rmtree(csv_dir)
    os.makedirs(csv_dir)
    data.to_csv(data_dict["save_csv_file"], index=False)
                

def generalizability_plot(data_dict):
    """
    Plot the generalizability of parameters between different weekend IDs. 
    """
    csv_file = data_dict["save_csv_file"]
    data = pd.read_csv(csv_file)
    ids = [1,2,3,5,6,6.5,7,7.5,8,9,10]
    cols = [col for col in data.columns if col.startswith("Quantile")]
    fig, axs = plt.subplots(1, len(cols), figsize=(16, 4))
    img_folder = os.path.dirname(csv_file)
    for ic, col in enumerate(cols):
        mat = np.zeros((len(ids), len(ids)))
        for i, train_id in enumerate(ids):
            for j, val_id in enumerate(ids):
                row = data[data["Train ID"] == train_id][data["Val ID"] == val_id]
                assert len(row) == 1
                mat[i,j] = row[col].iloc[0] / row["Val length"].iloc[0]
        ax = axs[ic]
        ax.matshow(mat)
        ax.set_title(f"Gen. Plot {col}")
        ax.set_xlabel("Train")
        ax.set_ylabel("Val")
        ax.set_xticks(np.arange(len(ids)), ids)
        ax.set_yticks(np.arange(len(ids)), ids)
    img_file = os.path.join(img_folder, f"Generalizability Plot.png")
    
    normalizer = matplotlib.colors.Normalize(0,1)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer), ax=axs)
    fig.savefig(img_file)
    plt.close(fig)

if __name__ == "__main__":
    # create_csv()
    # check_run_id()
    # train_stable()
    # train_all_combinations()
    # train_generalizability()
    # save_trainer()
    # bar_plot_errors()
    # params_generalizability()
    # launch_train_generalizability()
    # generalizability_plot()
    pass
