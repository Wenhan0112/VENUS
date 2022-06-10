import torch
from torch import nn
import torch.optim
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import physical_model
import VENUS_dataset
import gaussian_process_regressor

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")


def visualize_val(trainers, part=20, labels=None, plot_comparison=False):
    num = len(trainers)
    [trainer.model.eval() for trainer in trainers]
    val_dataloaders = [trainer.val_dataloader for trainer in trainers]
    batchs = [next(iter(val_dataloader)) for val_dataloader in val_dataloaders]
    preds = [trainers[i].model(batchs[i]["input"]) for i in range(num)]
    preds = [pred.detach().cpu().flatten().numpy() for pred in preds]
    targets = [batch["output"].detach().cpu().flatten().numpy()
        for batch in batchs]
    errs = [preds[i] - targets[i] for i in range(num)]

    for i in range(num):
        label = labels[i] if labels else ""
        message = labels[i] if labels else f"Index {i}"
        print(f"Average Absolute Error for {message}", np.abs(errs[i]).mean())
        plt.hist(errs[i], bins="sqrt", label=label, density=True, alpha = 1 / num)
    plt.xlabel("Error")
    plt.ylabel("PDF")
    plt.title("Error Historgram")
    if labels:
        plt.legend()
    plt.show()

    if plot_comparison:
        pred = pred[:part]
        target = target[:part]
        size = len(pred)
        indices = np.arange(size)

        width = 0.3

        plt.bar(indices - width / 2, pred , width, label='Prediction')
        plt.bar(indices + width / 2, target, width, label='Ground Truth')
        plt.legend()
        plt.title("Value Comparison")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.show()

def train_nn():
    model = Naive_Net([6, 16, 32, 8, 1])
    # model = Naive_Net([6, 1])
    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    # input_cols = input_cols[:3]
    output_cols = ["beam_avg"]
    train_data, val_data = read_and_split_data(0.2)
    # train_data, val_data = read_and_split_data_by_trial()
    dataset = VENUS_Dataset(train_data, input_cols, output_cols)
    val_dataset = VENUS_Dataset(val_data, input_cols, output_cols)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
    trainer = Trainer(dataloader, model, device=device)
    trainer.set_validation(val_dataloader)
    trainer.train(200, 3e-4, if_print=True)
    plt.plot(trainer.val_losses)
    plt.yscale("log")
    plt.title("Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.show()
    visualize_val(model, val_dataset)

def train_val_by_trial():

    def train(train_data, val_data):
        model = Naive_Net([6, 32, 64, 64, 16, 1])
        dataset = VENUS_Dataset(train_data, input_cols, output_cols)
        val_dataset = VENUS_Dataset(val_data, input_cols, output_cols)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))
        trainer = Trainer(dataloader, model, device=device)
        trainer.set_validation(val_dataloader)
        trainer.train(80, 1e-3)
        return trainer

    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    output_cols = ["beam_avg"]
    train_data_trial, val_data_trial = read_and_split_data_by_trial()
    frac = len(val_data_trial) / (len(val_data_trial) + len(train_data_trial))
    train_data_rd, val_data_rd = read_and_split_data(frac)
    trainer_trial = train(train_data_trial, val_data_trial)
    trainer_rd = train(train_data_rd, val_data_rd)
    plt.plot(trainer_trial.val_losses, label="By Trial")
    plt.plot(trainer_rd.val_losses, label="Random")
    plt.yscale("log")
    plt.title("Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    visualize_val([trainer_trial, trainer_rd], labels=["By Trial", "Random"])
    return locals()



def find_kernel_length_scale():
    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    output_cols = ["beam_avg"]
    train_data, val_data = VENUS_dataset.read_and_split_data_by_trial()

    train_dataset = VENUS_dataset.VENUS_Dataset(
        train_data, input_cols, output_cols
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset))
    train_data = next(iter(train_dataloader))
    train_input, train_output = train_data["input"], train_data["output"]

    val_dataset = VENUS_dataset.VENUS_Dataset(
        val_data, input_cols, output_cols
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset))
    val_data = next(iter(val_dataloader))
    val_input, val_output = val_data["input"], val_data["output"]


    search_set = np.linspace(0.05, 5, 50)
    best_val, scores = gaussian_process_regressor.kernel_length_scale_search(
        search_set, train_input, train_output, cv=5
    )
    print("Best kernel length scale:", best_val)
    plt.plot(search_set, scores, "b.")
    # plt.xscale("log")
    plt.xlabel("Kernel length scale")
    plt.ylabel("Score")
    plt.show()



def train_nn_and_gp(datatype="random"):

    def train_nn(train_data, val_data):
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=12)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
        model = physical_model.Naive_Net_Model(dataloader,
            [len(input_cols), 32, 64, 128, 64, 16, 1], device=device)
        model.set_validation(val_dataloader)
        model.train(100, 1e-2)
        return model.val_losses, model

    def train_gpr(train_data, val_data):
        nu = 0.66
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
        train_data = next(iter(dataloader))
        input, output = train_data["input"], train_data["output"]
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
        train_data = next(iter(val_dataloader))
        val_input, val_output = train_data["input"], train_data["output"]
        model = gaussian_process_regressor.Gaussian_Process_Regressor(nu)
        model.train(input, output)
        return model.loss(val_input, val_output), model

    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    # input_cols = ["inj_avg", "ext_avg", "mid_avg"]
    output_cols = ["beam_avg"]
    if datatype == "random":
        trial = [3]
        datatype += f" trial {trial}"
        train_data, val_data = VENUS_dataset.read_and_split_data(0.8, trial)
    else:
        train_data, val_data = \
            VENUS_dataset.read_and_split_data_by_trial(None, datatype)
    dataset = VENUS_dataset.VENUS_Dataset(train_data, input_cols, output_cols)
    val_dataset = VENUS_dataset.VENUS_Dataset(val_data, input_cols, output_cols)
    nn_losses, nn_model = train_nn(dataset, val_dataset)
    gpr_loss, gpr_model = train_gpr(dataset, val_dataset)
    print("Minumum neural net loss:", np.array(nn_losses).min())
    print("GPR loss:", gpr_loss)
    plt.plot(nn_losses, marker="s", color="b", linestyle="None",
        label="Neural Net")
    plt.axhline(gpr_loss, color="r", label="GPR")
    plt.yscale("log")
    plt.title(f"Validation Losses (Datatype {datatype}, {len(input_cols)} vars)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    return locals()

def train_physical_nn(datatype="random"):

    def train_nn(train_data, val_data, loss, regularization):
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=12)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
        model = physical_model.Naive_Net_Model(dataloader,
            [len(input_cols), 32, 64, 128, 64, 16, 1],
            loss=loss, regularization=regularization, device=device)
        model.set_validation(val_dataloader)
        model.train(200, 2e-2)
        return model.val_losses, model.val_mse_losses, model

    def visualize_physical(train_data, val_data, model):
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=10)
        data = next(iter(val_dataloader))
        input, output = data["input"], data["output"]
        pred = model.predict(input).detach().numpy()
        x = np.arange(len(pred))
        plt.errorbar(x, pred[:, 0], yerr=pred[:, 1], label="Prediction",
            linestyle="None", capsize=5, marker="s", markersize=5)
        plt.errorbar(x, output[:, 0], yerr=output[:, 1], label="Validation",
            linestyle="None", capsize=5, marker="s", markersize=5)
        for i in x[:-1]:
            plt.axvline(i + 0.5)
        plt.xlabel("Index")
        plt.ylabel("Beam current ($\\mathrm{\\mu A}$)")
        plt.tick_params(direction="in")
        plt.title("Validation Visualization")
        plt.legend()
        plt.show()

    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    # input_cols = ["inj_avg", "ext_avg", "mid_avg"]
    output_cols = ["beam_avg", "beam_std"]
    if datatype == "random":
        trial = [0]
        datatype += f" trial {trial}"
        train_data, val_data = VENUS_dataset.read_and_split_data(0.8, trial)
    else:
        train_data, val_data = \
            VENUS_dataset.read_and_split_data_by_trial(None, datatype)
    dataset = VENUS_dataset.VENUS_Dataset(train_data, input_cols, output_cols)
    val_dataset = VENUS_dataset.VENUS_Dataset(val_data, input_cols, output_cols)
    nn_losses, mse_losses, model = train_nn(dataset, val_dataset,
        loss="Physical", regularization=1.)
    # physical_loss = train_nn(dataset, val_dataset)
    print("Minumum neural net loss:", np.array(nn_losses).min())
    dataset = VENUS_dataset.VENUS_Dataset(train_data, input_cols, output_cols)
    val_dataset = VENUS_dataset.VENUS_Dataset(val_data, input_cols, output_cols)
    visualize_physical(dataset, val_dataset, model)
    plt.plot(nn_losses, marker="s", color="b", linestyle="None",
        label="Neural Net")
    plt.yscale("log")
    plt.title(f"Validation Losses Physical Net\n(Val trial {datatype}, "
        f"{len(input_cols)} vars)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    plt.plot(mse_losses, marker="s", color="b", linestyle="None",
        label="Neural Net")
    plt.yscale("log")
    plt.title(f"Validation MSE Losses Physical Net\n(Val trial {datatype}, "
        f"{len(input_cols)} vars)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    return locals()

def train_bhattacharyya_nn(datatype="random"):

    def train_nn(train_data, val_data, loss, regularization):
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=12)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
        model = physical_model.Naive_Net_Model(dataloader,
            [len(input_cols), 32, 64, 128, 64, 16, 1],
            loss=loss, regularization=regularization, device=device)
        model.set_validation(val_dataloader)
        model.train(100, 2e-2)
        return model.val_losses, model

    def visualize_physical(data, model):

        input, output = data["input"], data["output"]
        pred = model.predict(input).detach().numpy()
        x = np.arange(len(pred))
        plt.errorbar(x, pred[:, 0], yerr=pred[:, 1], label="Prediction",
            linestyle="None", capsize=5, marker="s", markersize=5)
        plt.errorbar(x, output[:, 0], yerr=output[:, 1], label="Validation",
            linestyle="None", capsize=5, marker="s", markersize=5)
        for i in x[:-1]:
            plt.axvline(i + 0.5)
        plt.xlabel("Index")
        plt.ylabel("Beam current ($\\mathrm{\\mu A}$)")
        plt.tick_params(direction="in")
        plt.title("Validation Visualization")
        plt.legend()
        plt.show()

    input_cols = ["inj_avg", "ext_avg", "mid_avg", "bias_avg", "ext_p_avg", "inj_p_avg"]
    # input_cols = ["inj_avg", "ext_avg", "mid_avg"]
    output_cols = ["beam_avg", "beam_std"]
    if datatype == "random":
        train_data, val_data = VENUS_dataset.read_and_split_data(0.8)
    else:
        train_data, val_data = \
            VENUS_dataset.read_and_split_data_by_trial(None, datatype)
    dataset = VENUS_dataset.VENUS_Dataset(train_data, input_cols, output_cols)
    val_dataset = VENUS_dataset.VENUS_Dataset(val_data, input_cols, output_cols)
    # nn_losses_1, model_1 = train_nn(dataset, val_dataset,
    #     loss="Physical", regularization=1.)
    # physical_loss = train_nn(dataset, val_dataset)
    nn_losses, model = train_nn(dataset, val_dataset,
        loss="Bhattacharyya", regularization=None)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10)
    val_data = next(iter(val_dataloader))
    print("Minumum neural net loss:", np.array(nn_losses).min())
    visualize_physical(val_data, model)
    plt.plot(nn_losses, marker="s", color="b", linestyle="None",
        label="Neural Net")
    plt.yscale("log")
    plt.title(f"Validation Losses Physical Net\n(Val trial {datatype}, "
        f"{len(input_cols)} vars)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()
    return locals()

if __name__ == "__main__":
    # data = train_val_by_trial()
    # find_kernel_length_scale()
    # data = train_nn_and_gp()
    data = train_physical_nn("random")
    # data = train_bhattacharyya_nn("random")
