"""
The trainer wrapper around the neural network. 
    Date: May 2023
    Author: Wenhan Sun
"""

import torch
from torch import nn
import torch.optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import math
import os
import pickle
import physical_model
import shutil
from typing import Optional, Callable

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

class Trainer():
    """
    A trainer that wraps around the core neural net class for analysis.

    @field dataset (torch.utils.data.Dataset): The dataset used to train the 
        neural net. 
    @field loss_fns (list[torch.nn.Module]): A list of loss function used. 
    @field model (torch.nn.Module): The core neural net.
    @field truth_processor (torch.nn.Module): The function that the truth 
        passes through before comparing with the predictions in the loss 
        functions. 
    @field loss_integration (Callable[[list[torch.Tensor]], torch.Tensor]): 
        A function that takes a list of losses computed by different loss 
        functions and output the loss derived from an integration of different 
        loss functions. 
    @field validation (bool): True iff validation is performed.
    @field device (torch.device): The device where the parameters are stored. 
    @field lr_scheduler_setter (Optional[
            Callable[[torch.optim.Optimizer], torch.optim.LRScheduler]
        ]): A function to take the learning rate schedular. It takes in an 
        optimizer and outputs the learning rate scheduler that is linked to 
        this optimizer. If not specified, then no learning rate scheduler is 
        used. 
    @field lr_scheduler (torch.optim.LRScheduler): The learning rate 
        scheduler. If not specified, then no learning rate scheduler is used. 
    """
    def __init__(self, dataset: torch.utils.data.Dataset, model: nn.Module, 
            loss_fns: list[str|nn.Module] = ["MSE"], 
            device: torch.device=device, *,
            standardizer: Optional[physical_model.Data_Transformer] = None, 
            truth_processor: Optional[nn.Module] = None, 
            loss_integration: 
            Optional[Callable[[list[torch.Tensor]], torch.Tensor]] = None,
            lr_scheduler_setter: 
            Optional[
                Callable[[torch.optim.Optimizer], 
                    torch.optim.lr_scheduler.LRScheduler]
            ] = None):
        """
        Constructor. 
        @params dataset (torch.utils.data.Dataset): The dataset used to train 
            the neural net. 
        @parans model (torch.nn.Module): The core neural net.
        @params loss_fns (list[str|torch.nn.Module]): A list of loss function 
            used. If `"MSE"` is included, it is converted to MSE loss
            `torch.nn.MSELoss`. If `"MAE"` is included, it is converted to MAE 
            loss `torch.nn.L1Loss`. If `"Rel_MSE"` is included, it is 
            converted to relative MSE loss. If `"Rel_MAE"` is included, it is 
            converted to relative MAE loss. 
            CONSTRAINT: [isinstance(l, torch.nn.Module) 
                or l in ["MAE", "MSE", "Rel_MAE", "Rel_MSE"] 
                for l in loss_fns]
        @params device (torch.device): The device where the parameters are 
            stored. 
        @params standardizer (Optional[physical_model.Data_Transformer], 
            DEFAULT None): The standardizer used to transform the dataset. If 
            not specified, the identity transformation is used, that is, no 
            transformation.
        @params truth_processor (Optional[torch.nn.Module], DEFAULT None): The 
            function that the truth passes through before comparing with the 
            predictions in the loss functions. If not specified, the the 
            identity transformation is used, that is, no transformation.
        @params loss_integration 
            (Optional[Callable[[list[torch.Tensor]], torch.Tensor]], 
            DEFAULT None): A function that takes a list of losses computed by 
            different loss functions and output the loss derived from an 
            integration of different loss functions. If not specified, then 
            the zeroth loss function in `loss_fns` is used. Others are 
            neglected during optimizations. The length of the input list is 
            expected to the same as the length of `loss_fns`, where the i-th 
            entry is expected to corresponds to the loss outputed by the i-th 
            loss function. 
        @params lr_scheduler_setter (Optional[
                Callable[[torch.optim.Optimizer], 
                    torch.optim.lr_scheduler.LRScheduler]
            ]): A function to take the learning rate schedular. It takes in 
            an optimizer and outputs the learning rate scheduler that is 
            linked to this optimizer. If not specified, then no learning rate 
            scheduler is used. 
        """
        self.dataset = dataset
        self.loss_fns = [self.convert_loss_fn(loss_fn) for loss_fn in loss_fns]
        self.model = model

        # Set truth processor
        if truth_processor:
            self.truth_processor = truth_processor
        else:
            self.truth_processor = nn.Identity()

        # Set standardizer
        if standardizer:
            self.standardizer = standardizer
        else:
            self.standardizer = physical_model.Identity_Transformer()

        # Set loss integration
        if loss_integration is None:
            self.loss_integration = lambda x: x[0]
        else:
            self.loss_integration = loss_integration
        

        self.validation = False
        self.device = device
        self.lr_scheduler_setter = lr_scheduler_setter
        self.lr_scheduler = None

    
    def set_standardizer(self, standardizer, standardizer_params=[], standardizer_kwargs={}):
        """
        Deprecated!
        """
        raise DeprecationWarning("This method is deprecated! Initialize the " 
            "trainer directly with the desired standardizer. ")
        if standardizer == "Naive":
            self.standardizer = physical_model.Naive_Standardizer(device=device, **standardizer_kwargs)
        elif standardizer == "Partial":
            self.standardizer = physical_model.Partial_Standardizer(device=device, **standardizer_kwargs)
        elif standardizer == "PCA":
            self.standardizer = physical_model.PCA(*standardizer_params, **standardizer_kwargs)
        elif standardizer == None:
            self.standardizer = physical_model.Identity_Transformer(device=device, **standardizer_kwargs)
        else:
            raise ValueError(f"Undefined standardizer: {standardizer}")
    
    @staticmethod
    def convert_loss_fn(loss_fn: str|nn.Module) -> nn.Module:
        """
        Convert a loss function, possibly a string, to a Pytorch module. 
        @params loss_fn (str|torch.nn.Module): A loss function, possibly a 
            string, to convert. `"MSE"` is converted to MSE loss 
            `torch.nn.MSELoss`. `"MAE"` is converted to MAE loss 
            `torch.nn.L1Loss`. `"Rel_MSE"` is converted to relative MSE loss. 
            `"Rel_MAE"` is converted to relative MAE loss. 
            CONSTRAINT: isinstance(loss_fn, torch.nn.Module) \
                or loss_fn in ["MAE", "MSE", "Rel_MAE", "Rel_MSE"]
        @return (torch.nn.Module): The converted Pytorch Module. 
        @error (ValueError): Error if `loss_fn` is a string but does not 
            belong to `"MAE"`, `"MSE"`, `"Rel_MAE"`, `"Rel_MSE"`. 
        """
        if loss_fn == "MSE":
            return nn.MSELoss()
        elif loss_fn == "MAE":
            return nn.L1Loss()
        elif loss_fn == "Rel_MSE":
            return physical_model.Rel_Loss("MSE")
        elif loss_fn == "Rel_MAE":
            return physical_model.Rel_Loss("MAE")
        elif isinstance(loss_fn, nn.Module):
            return loss_fn
        else:
            raise ValueError(f"{loss_fn} not implemented as a loss function!")

    def set_validation(self, 
            val_dataset: torch.utils.data.Dataset|torch.utils.data.DataLoader, 
            val_batchsize: int) -> None:
        """
        Set up validation dataset. 
        @params val_dataset 
            (torch.utils.data.Dataset|torch.utils.data.DataLoader): 
            The validation dataset or data loader.
        @params val_batchsize: The validation dataset loader batch size. If it 
            is 0, then `val_dataset` is interpreted as a dataloader. If it is 
            smaller than 0, the entire validation dataset is loaded at the same 
            time.
        """
        self.validation = True
        self.val_dataset = val_dataset
        if val_batchsize < 0:
            val_batchsize = len(val_dataset)
        if val_batchsize:
            self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                val_batchsize)
        else:
            self.val_dataloader = val_dataset


    def load_model(self, model_file: str) -> None:
        """
        Load the model from the state dictionary file.
        @params model_file (str): The path to the file containing the state 
            dictionary of the model.
        """
        self.model.load_state_dict(model_file)
    
    def set_lr_scheduler(self, lr_scheduler_class, lr_scheduler_kwargs={}):
        """
        Deprecated! 
        """
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def train(self, num_epochs: int, lr: float, train_batchsize: int, *,
            verbose: int = 0, save_model=None, save_model_file=None):
        """
        Launch the training process.
        @params num_epochs (int): The number of epochs.
            CONSTRAINT: num_epochs > 0
        @params lr (float): Initial learning rate. 
            CONSTRAINT: lr > 0
        @params train_batchsize: The validation dataset loader batch size. If 
            it is 0, then `self.dataset` is interpreted as a dataloader. If it 
            is smaller than 0, the entire training dataset is loaded at the 
            same time.
        @params verbose (int, DEFAULT 0): Verbose index. 
            0: Nothing
            1: Print without tqdm bars
            2: Print with tqdm bars
            CONSTRAINT: verbose > 0
        @params save_model (Optional[str]): The string to indicate which model 
            to save.
            `None`: No model is saved.
            `"best"`: The best model is saved based on validation loss. 
            Validation must be set up to use this feature.
            `"last"`: The model from the last epoch.
            CONSTRAINT: save_model in (None, "best", "last")
            CONSTRAINT: (save_model != "best") or self.validation
        @error (ValueError): Error if `save_model` is not one of `None`, 
            `"best"`, `"last"`.
        @error (ValueError): Error is model is save based on the best 
            validation loss without setting up validation dataset. 
        """
        if save_model not in (None, "best", "last"):
            raise ValueError("`save_model` must be one of `None`, `'best'`, " 
                f"`'last'`, but it is {save_model}!")
        if (save_model == "best") and not self.validation:
            raise ValueError("Selection of the best model based on "
                "validation loss must have validation dataset set up!")
        self.model.train()
        self.train_losses = [[] for _ in self.loss_fns]
        num_loss_fns = len(self.loss_fns)
        if self.validation:
            self.val_losses = [[] for _ in self.loss_fns]
        self.standardizer.fit(self.dataset, train_batchsize)

        # Set dataloader
        if train_batchsize < 0:
            train_batchsize = len(self.dataset)
        if train_batchsize:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                train_batchsize, shuffle=True)
        else:
            self.dataloader = self.dataset

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.lr_scheduler_setter:
            self.lr_scheduler = self.lr_scheduler_setter(self.optimizer)

        # Helper function to call when the model needs to be validated. 
        def validate():
            if not self.validation:
                return
            self.model.eval()
            epoch_val_losses = [[] for _ in self.loss_fns]
            weights = []
            for batch in self.val_dataloader:
                batch = self.standardizer.transform(batch)
                val_pred = self.model(batch["input"])
                val_truth = self.truth_processor(batch["output"])
                val_losses = [loss_fn(val_pred, val_truth) for loss_fn in self.loss_fns]
                val_losses = [val_loss.detach().cpu().item() for val_loss in val_losses]
                if any(math.isnan(val_loss) for val_loss in val_losses):
                    print("One of val losses is nan")
                [epoch_val_losses[i].append(val_losses[i]) for i in range(num_loss_fns)]
                weights.append(batch["output"].shape[0])
            [self.val_losses[i].append(np.average(epoch_val_losses[i], weights=weights)) 
                for i in range(num_loss_fns)]
            if verbose:
                print("Validation loss:", [self.val_losses[i][-1] for i in range(num_loss_fns)])

            # Save the model if the validation loss is minimum when needed. 
            if save_model == "best":
                if self.val_losses[-1] == np.min(self.val_losses):
                    torch.save(self.model.state_dict(), save_model_file)
            self.model.train()

        validate()

        for epoch in range(num_epochs):
            if verbose:
                print(f"Training epoch {epoch}")
            batch_iterator = enumerate(self.dataloader)
            if verbose >= 2:
                batch_iterator = tqdm.tqdm(batch_iterator,
                    total=len(self.dataloader))
            epoch_losses = [[] for _ in self.loss_fns]
            weights = []
            for i, batch in batch_iterator:
                self.optimizer.zero_grad()
                batch = self.standardizer.transform(batch)
                pred = self.model(batch["input"])
                truth = self.truth_processor(batch["output"])
                losses = [loss_fn(pred, truth) for loss_fn in self.loss_fns]
                self.loss_integration(losses).backward()
                self.optimizer.step()
                losses = [loss.detach().cpu().item() for loss in losses]
                [epoch_losses[i].append(losses[i]) for i in range(num_loss_fns)]
                weights.append(batch["output"].shape[0])
            [self.train_losses[i].append(np.average(epoch_losses[i], weights=weights)) 
                for i in range(num_loss_fns)]
            if verbose:
                print("Training loss", [self.train_losses[i][-1] for i in range(num_loss_fns)])

            validate()
            # Step the learning rate scheduler. 
            if self.validation and self.lr_scheduler:
                self.lr_scheduler.step(self.loss_integration([val_losses[-1] for val_losses in self.val_losses]))
            elif self.lr_scheduler:
                self.lr_scheduler.step()

        # Save the last model when needed. 
        if save_model == "last":
            torch.save(self.model.state_dict(), save_model_file)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """
        Predict the output based on the given input.
        @params input (torch.Tensor): The input data.
        @return (torch.Tensor): The output data. 
        """
        self.model.eval()
        input = self.standardizer.transform_input(input)
        output = self.model(input)
        output = self.standardizer.inv_transform_output(output)
        return output
    
    def get_data(self, dataset: str|torch.utils.data.Dataset) -> torch.Tensor: 
        """
        Get data from the trainer.
        @params dataset (str|torch.utils.data.Dataset): The dataset from which 
            the data is got.
            `"train"`: The training dataset.
            `"validation"` or `"val"`: The validation dataset.
            CONSTRAINT: dataset in ["train", "validation", "val"] \
                or isinstance(dataset, torch.utils.data.Dataset)
        @return (torch.Tensor): The data from the desired dataset. 
        @error (ValueError): Error if `dataset` is not `"train"`, 
                `"validation"`, `"val"`, or a 'torch.utils.data.Dataset' 
                object. 
        @error (ValueError): Error if validation dataset is wanted to be 
        retrieved but not specified. 
        """
        if dataset == "train":
            dataset = self.dataset
        elif dataset in ["validation", "val"]:
            if not self.validation:
                raise ValueError("Validation dataset is not specified. "
                    "Cannot get validation dataset!")
            dataset = self.val_dataset
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("`dataset` must be `'train'`, `'validation'`, "
                "`'val'`, or a 'torch.utils.data.Dataset' object, but it is "
                f"{dataset}")
        return next(iter(torch.utils.data.DataLoader(dataset, len(dataset))))
    
    def get_err(self, dataset: str|torch.utils.data.Dataset, 
            err_type: str = "abs") \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the data, error, prediction, and truth of a dataset.
        @params dataset (str|torch.utils.data.Dataset): The dataset from which 
            the data is got.
            `"train"`: The training dataset.
            `"validation"` or `"val"`: The validation dataset.
            CONSTRAINT: dataset in ["train", "validation", "val"] \
                or isinstance(dataset, torch.utils.data.Dataset)
        @params error_type (str, DEFAULT "abs"): The error type. 
            `"abs"`: Absolute error.
            `"rel"`: Relative error. 
            CONSTRAINT: err_type in ["abs", "rel"]
        @return 
            (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
            The data, error, prediction, and truth in sequence. 
        @error (ValueError): Error if `err_type` is not one of `"abs"` or 
            `"rel"`. 
        """
        data = self.get_data(dataset)
        pred = self.predict(data["input"]).detach()
        truth = self.truth_processor(data["output"])
        if err_type == "abs":
            err = pred - truth 
        elif err_type == "rel":
            err = (pred - truth) / truth
        else:
            raise ValueError("`err_type` must be one of `'abs'`, `'rel'`, "
                f"but it is {err_type}")
        err = err.detach().cpu()
        return data, err, pred.detach().cpu(), truth.detach().cpu()


    def err_histogram(self, dataset: str|torch.utils.data.Dataset, 
            err_type: str = "abs", title: str = "", 
            savefig: str = None) -> None:
        """
        Plot the error histogram. 
        @params dataset (str|torch.utils.data.Dataset): The dataset from which 
            the data is got.
            `"train"`: The training dataset.
            `"validation"` or `"val"`: The validation dataset.
        @params error_type (str, DEFAULT "abs"): The error type. 
            `"abs"`: Absolute error.
            `"rel"`: Relative error. 
            CONSTRAINT: err_type in ["abs", "rel"]
        @params title (str): The title of the plot, appended after 
            `"Error histogram"`.
        @params savefig (str): The file name to save the figure. If it is 
            empty, the figure is not saved. 
        """
        _, err, _, _ = self.get_err(dataset, err_type)
        plt.hist(err.numpy(), bins="sqrt")
        plt.xlabel("Error")
        plt.ylabel("Count")
        plt.title(f"Error histogram {title.title()}")
        plt.tick_params(direction="in")
        if savefig:
            plt.savefig(savefig)
        plt.show()
        plt.close()
        return err
    
    def err_plot_truth(self, datasets: list[str|torch.utils.data.Dataset], 
            err_type: str = "abs", title: str = "", 
            legends: Optional[list[str]] = None, savefig: str = ""):
        """
        Scatter plot the error of prediction and the truth of the datasets. 
        @params datasets (list[str|torch.utils.data.Dataset]): The list of 
            dataset from which the data is got. If each entry is: 
            `"train"`: The training dataset.
            `"validation"` or `"val"`: The validation dataset.
            CONSTRAINT: all([d in ["train", "validation", "val"] 
                or isinstance(d, torch.utils.data.Dataset) for d in datasets]) 
        @params error_type (str, DEFAULT "abs"): The error type. 
            `"abs"`: Absolute error.
            `"rel"`: Relative error. 
            CONSTRAINT: err_type in ["abs", "rel"]
        @params title (str): The title of the plot, appended after 
            `"Error Truth Plot"`
        @params legends (Optional[list[str]]): A list of legends in order with 
            `datasets`. If not specified, then `[str(d) for d in datasets]` is
            used. 
        @params savefig (str): The file name to save the figure. If it is 
            empty, the figure is not saved. 
        """
        unit = " ($\mu$A)" if err_type == "abs" else "" # Setup the correct unit
        if not legends:
            legends = datasets
        if title:
            title = " " + title
        truths = []
        errs = []
        for dataset in datasets:
            _, err, _, truth = self.get_err(dataset, err_type)
            truths.append(truth)
            errs.append(err)
        for truth, err, label in zip(truths, errs, legends):
            plt.scatter(truth.numpy(), err.numpy(), s=1, label=str(label).title())
        plt.xlabel("Actual Beam Current ($\mu$A)")
        plt.ylabel(f"{err_type.title()} Error{unit}")
        plt.title(f"{err_type.title()} Error Truth Plot{title}")
        if err_type == "rel" and err.abs().max() > 1:
            plt.yscale("symlog")
        plt.legend()
        plt.tick_params(direction="in")
        if savefig:
            plt.savefig(savefig)
        plt.show()
        plt.close()
    
    def err_plot_time(self, datasets: list[str|torch.utils.data.Dataset], 
            err_type: str = "abs", title: str = "", 
            legends: Optional[list[str]] = None, savefig: str = ""):
        """
        Scatter plot the error of prediction and the time of the datasets. 
        @params datasets (list[str|torch.utils.data.Dataset]): The list of 
            dataset from which the data is got. If each entry is: 
            `"train"`: The training dataset.
            `"validation"` or `"val"`: The validation dataset.
            CONSTRAINT: all([d in ["train", "validation", "val"] 
                or isinstance(d, torch.utils.data.Dataset) for d in datasets]) 
        @params error_type (str, DEFAULT "abs"): The error type. 
            `"abs"`: Absolute error.
            `"rel"`: Relative error. 
            CONSTRAINT: err_type in ["abs", "rel"]
        @params title (str): The title of the plot, appended after 
            `"Error Truth Plot"`
        @params legends (Optional[list[str]]): A list of legends in order with 
            `datasets`. If not specified, then `[str(d) for d in datasets]` is
            used. 
        @params savefig (str): The file name to save the figure. If it is 
            empty, the figure is not saved. 
        """
        unit = " ($\mu$A)" if err_type == "abs" else ""
        if not legends:
            legends = datasets
        if title:
            title = " " + title
        datas = []
        errs = []
        for dataset in datasets:
            data, err, _, _ = self.get_err(dataset, err_type)
            datas.append(data)
            errs.append(err)
        for data, err, label in zip(datas, errs, legends):
            plt.scatter(data["time"].cpu().numpy(), err.numpy(), s=1, label=str(label).title())
        plt.xlabel("Unix Epoch Time (ms)")
        plt.ylabel(f"{err_type.title()} Error{unit}")
        if err_type == "rel" and err.abs().max() > 1:
            plt.yscale("symlog")
        plt.title(f"{err_type.title()} Error Time Plot{title}")
        plt.legend()
        plt.tick_params(direction="in")
        if savefig:
            plt.savefig(savefig)
        plt.show()
        plt.close()
    
    
    def save_model(self, folder: str, cols_dict: Optional[dict] = None):
        """
        Save the model and associated files to a folder. 
        There are three files in the folder. 
        `"accessory"`: a pickled file with all the supplemental transforms. 
        `"model"`: a binary file of the model compatible with Pytorch.
        `"password.txt"`: A password file with password only: 
            `"VERY_NAIVE_VENUS"`. 
        @params folder (str): The folder in which the files are. If 
            the folder already exists, the content is deleted. 
        @params cols_dict (Optional[dict]): The columns dictionary of that the 
            model would use. If not specified, it is the columns dictionary 
            associated with the dataset.
            cols_dict["input"]: The input columns
            cols_dict["output"]: The output columns
            CONSTRAINT: cols_dict is None or \
                ["input", "output"] in cols_dict.keys()
        """
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        accessory_fname = os.path.join(folder, "accessory")
        model_fname = os.path.join(folder, "model")
        password_fname = os.path.join(folder, "password.txt")

        # Accessory file
        accessory_file = open(accessory_fname, "wb")
        save_dict = {}
        save_dict["standardizer"] = self.standardizer
        cols_dict = cols_dict if cols_dict else self.dataset.cols_dict
        save_dict["cols"] = {"input":cols_dict["input"], "output":cols_dict["output"]}
        pickle.dump(save_dict, accessory_file)
        accessory_file.close()

        # Model file
        torch.save(self.model, model_fname)

        # Password file
        password_file = open(password_fname, "w")
        password_file.write("VERY_NAIVE_VENUS")
        password_file.close()
        

       
        
