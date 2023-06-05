"""
The neural network models used on VENUS dataset.
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
from typing import Optional

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

class Scaler(nn.Module):
    """
    A linear scaler layer. 

    @field scale (float): The scaling factor
    """
    def __init__(self, scale: float):
        """
        Constructor.
        @params scale (float): The scaling factor
        """
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        """
        OVERRIDE
        """
        return x * self.scale
    
    def inverse(self):
        """
        Return the inverse function of this scaler, that is, the scaler with 
            scaling factor that is the reciprocal of the original.
        @return (Scaler): The inverse function of this scaler. 
        @error (ValueError): Error if the original scaling factor is 0, where 
            no inverse exists. 
        """
        if self.scale == 0:
            raise ValueError("The inverse does not exists if the scaling " 
                "factor is 0!")
        return Scaler(1 / self.scale)

class Naive_Net(nn.Module):
    """
    A Naive neural network, just a multi-layer perceptron. 
        The initial input passes through a batch-norm layer, while after 
        each linear layer, the output also passes through a batch-norm layer 
        except for the last. The activation function is GELU. 

    @field device (torch.device): The device where the parameters are stored. 
    @field layers (nn.ModuleList): A list of layers in sequence. 
    """
    def __init__(self, layer_sizes: list[int], device=device):
        """
        Constructor.
        @params layer_sizes (list[int]): The input, hidden, and output layer 
            dimensions. `layer_sizes[0]` is the input dimension. 
            `layer_sizes[-1]` is the output dimensions. The rest are the 
            intermediate dimensions in sequence.
            CONSTRAINT: len(layer_sizes) > 1
            CONSTRAINT: all([s > 0 for s in layer_sizes])
        @params device (torch.device): The device where the parameters are 
            stored. 
        @error (ValueError): Error if there are less than 2 layers. 
        @error (ValueError): Error if there are layers with non-positive 
            dimensions. 
        """
        if len(layer_sizes) <= 1:
            raise ValueError("There must be more than 2 layers!")
        if not all([s > 0 for s in layer_sizes]):
            raise ValueError("All dimensions must be positive!")
        super(Naive_Net, self).__init__()
        self.device=device
        layers = [nn.BatchNorm1d(layer_sizes[0], device=device)]
        i = -1
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1],
                device=device))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1], device=device))
            layers.append(nn.GELU())
        layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i+2],
            device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        OVERRIDE
        @params x (torch.Tensor): Model input. `x.ndim == 2` and the first 
            dimension must be the input dimension. 
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Fully_Conv_Net(nn.Module):
    """
    A fully convolutional neural network. The input is a single datapoint. 
    The 
    The initial input passes through a 1-dimensional batch-norm layer, 
    treating the convolutional dimension as batch dimension. Then it 
    iteratively passes through the convolutional layer, the 1-dimensional 
    batch-norm layer, and the GELU activation function. After the last 
    convolutional layer, no batch-norm or activation is performed.  

    @field device (torch.device): The device where the parameters are stored. 
    @field norm_layers (nn.ModuleList): A list of batch-norm layers in 
        sequence. 
    @field conv_layers (nn.ModuleList): A list of convolutional layers in 
        sequence. 
    @field displacement (int): The convolutional dimension length reduction 
        due to convolution. 
    """
    def __init__(self, layer_sizes, kernel_sizes, device=device):
        """
        Constructor.
        @params layer_sizes (list[int]): The input, hidden, and output layer 
            dimensions. `layer_sizes[0]` is the input dimension. 
            `layer_sizes[-1]` is the output dimensions. The rest are the 
            intermediate dimensions in sequence.
            CONSTRAINT: len(layer_sizes) > 1
            CONSTRAINT: all([s > 0 for s in layer_sizes])
        @params kernel_sizes (list[int]): The kernel sizes for each 
            convolutional layer in sequence.
            CONSTRAINT: len(layer_sizes) == len(kernel_sizes) + 1
            CONSTRAINT: all([s > 0 for s in kernel_sizes])
        @params device (torch.device): The device where the parameters are 
            stored. 
        @error (ValueError): Error if there are less than 2 layers. 
        @error (ValueError): Error if there are layers with non-positive 
            dimensions.
        @error (ValueError): Error if the number of kernel sizes is not one 
            less than that of the layer sizes. 
        @error (ValueError): Error if there are non-positive kernel sizes.
        """
        if len(layer_sizes) <= 1:
            raise ValueError("There must be more than 2 layers!")
        if not all([s > 0 for s in layer_sizes]):
            raise ValueError("All dimensions must be positive!")
        if len(layer_sizes) != len(kernel_sizes) + 1:
            raise ValueError("The number of kernel sizes " 
                f"{len(kernel_sizes)} is not 1 less than the number of layer " 
                f"sizes {len(layer_sizes)}!")
        if not all([s > 0 for s in kernel_sizes]):
            raise ValueError("All kernel sizes must be positive!")
        super().__init__()
        self.device=device
        norm_layers = []
        conv_layers = []
        for i in range(len(layer_sizes) - 1):
            norm_layers.append(nn.BatchNorm1d(layer_sizes[i], device=device))
            conv_layers.append(
                nn.Conv1d(layer_sizes[i], layer_sizes[i+1], kernel_sizes[i],
                    device=device, padding="valid")
            )
        self.norm_layers = nn.ModuleList(norm_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.displacement = int(sum(kernel_sizes) - len(kernel_sizes))

    def forward(self, x):
        """
        OVERRIDE
        @params x (torch.Tensor): Model input. `x.ndim == 2` and the first 
            dimension must be the input dimension. 
        """
        for i in range(len(self.norm_layers)):
            x = self.norm_layers[i](x)
            if i != 0:
                x = nn.functional.gelu(x)
            x = x.transpose(0, 1).unsqueeze(0)
            x = self.conv_layers[i](x)
            x = x.squeeze(0).transpose(0, 1)
        return x

class Bhattacharyya_Loss(nn.Module):
    """
    The Bhattacharyya loss function between parameter of normal 
        distributions. 

    Eq.1 
    $$
    D(X, Y) = \frac{(\mu_X-\mu_Y)^2}{4(\simga_X^2+\sigma_Y^2)} + \frac{1}{2}
    \ln\left(\frac{\sigma_X^2+\sigma_Y^2}{2\sigma_Xsigma_Y}\right)
    $$

    """
    def forward(self, x, y):
        """
        OVERRIDE
        @params x (torch.Tensor): One input tensor. 
            CONSTRAINT: x.ndim == 2 and x.shape[1] == 2
            x[i, 0]: The expectation $\mu_X$.
            x[i, 1]: The standard deviation $\sigma_X$. 
        @params y (torch.Tensor): Another input tensor. 
            CONSTRAINT: y.shape == x.shape
            y[i, 0]: The expectation $\mu_Y$.
            y[i, 1]: The standard deviation $\sigma_Y$. 
        @return (torch.Tensor): The Bhattacharyya loss multiplied by a factor 
            of 4. 
            CONSTRAINT: RETURN.shape == torch.Size([])
        """
        x, sx = x[:, 0], x[:, 1]
        y, sy = y[:, 0], y[:, 1]
        sx2, sy2 = sx**2, sy**2
        return ((x - y)**2 / (sx2 + sy2)).sum() + \
            2 * torch.log((sx2 + sy2) / (2 * sx.abs() * sy.abs())).sum()

class Physical_Loss(nn.Module):
    """
    A physical loss function including the standard error of predictions.

    Eq.2
    $$
    D(X, Y) = \frac{(\mu_X-\mu_Y)^2}{\sigma_X^2+\sigma_Y^2} + 
    \lambda (\sigma_X^2+\sigma_Y^2)
    $$

    @field regularization (float): The regularization constant $\lambda$ in 
        Eq.2
    """
    def __init__(self, regularization: float):
        """
        Constructor.
        @params reuglarization (float): The regularization constant $\lambda$ 
            in Eq.2. 
        """
        super(Physical_Loss, self).__init__()
        self.regularization = regularization

    def forward(self, x, y):
        """
        OVERRIDE
        @params x (torch.Tensor): One input tensor. 
            CONSTRAINT: x.ndim == 2 and x.shape[1] == 2
            x[i, 0]: The expectation $\mu_X$.
            x[i, 1]: The standard error $\sigma_X$. 
        @params y (torch.Tensor): Another input tensor. 
            CONSTRAINT: y.shape == x.shape
            y[i, 0]: The expectation $\mu_Y$.
            y[i, 1]: The standard error $\sigma_Y$. 
        @return (torch.Tensor): The physical loss. 
            CONSTRAINT: RETURN.shape == torch.Size([])
        """
        x, sx = x[:, 0], x[:, 1]
        y, sy = y[:, 0], y[:, 1]
        sx2, sy2 = sx**2, sy**2
        return ((x - y)**2 / (sx2 + sy2)).sum() + \
            self.regularization * sx2.sum()

class Rel_Loss(nn.Module):
    """
    Loss functions with relative error.

    Eq.3
    $$
    D_r(x, y) = D(x/y, 1)
    $$

    @field loss (nn.Module): The base loss function. It has to take in two 
        torch.Tensor objects of the same shape.
    """
    def __init__(self, loss: str|nn.Module):
        """
        Constructor.
        @params loss (str|nn.Module): The base loss function. 
            If loss is `"MSE"`, then `torch.nn.MSELoss()` is used.
            If loss is `"MAE"`, then `torch.nn.L1Loss()` is used. 
            If loss is other strings, an error is raised.
            CONSTRAINT: loss in ["MSE", "MAE"] or isinstance(loss, nn.Module)
        @error (ValueError): Error is `loss` is a string other than 
            `"MSE"` or `"MAE"`.
        """
        super().__init__()
        if loss == "MSE":
            self.loss = nn.MSELoss()
        elif loss == "MAE":
            self.loss = nn.L1Loss()
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            raise ValueError(f"{loss} not implemented as a loss function!")
    
    def forward(self, x, y):
        """
        OVERRIDE
        @params x (torch.Tensor): One input to the loss function. 
        @params y (torch.Tensor): Another input to the loss function.
            CONSTRAINT: x.shape == y.shape
        """
        return self.loss(x/y, torch.ones_like(x))

class Data_Transformer():
    """
    Dataset transformmer abstract class. 

    @field device (torch.device): The device where the parameters are stored. 
    """
    def __init__(self, device=device):
        """
        Constructor.
        @params device (torch.device): The device where the parameters are 
            stored. Default is CUDA if CUDA is avaiable and CPU if not.  
        """
        self.device = device

    def fit(self, dataset: torch.utils.data.Dataset, 
            batchsize: Optional[int] = None) -> None:
        """
        ABSTRACT METHOD
        Fit the data transformer. 

        @params dataset (torch.utsil.data.Dataset): The dataset to fit the transformer. 
        @params batchsize (Optional[int]): The batch size used to extract 
            the dataset. 
        @error (NotImplementedError): Error if the method is not overriden. 
        """
        raise NotImplementedError("fit is not Not implemented!")

    def transform(self, batch: dict) -> dict:
        """
        Transform batch of the data. 

        @params batch (dict): The batch of data.
            It must have at least two entries, `"input"` and `"output"`, as 
            the input data and ouput data respectively. All other entries are 
            not modified. 
            CONSTRAINT: "input" in batch and "output" in batch
        @return (dict): The transformed batch of data. 
        """
        batch = dict(batch)
        batch["input"] = self.transform_input(batch["input"])
        batch["output"] = self.transform_output(batch["output"])
        return batch
    
    def inv_transform(self, batch: dict) -> dict:
        """
        Inverse transform batch of the data. 

        @params batch (dict): The batch of data.
            It must have at least two entries, `"input"` and `"output"`, as 
            the input data and ouput data respectively. All other entries are 
            not modified. 
            CONSTRAINT: "input" in batch and "output" in batch
        @return (dict): The inverse transformed batch of data. 
        """
        batch = dict(batch)
        batch["input"] = self.inv_transform_input(batch["input"])
        batch["output"] = self.inv_transform_output(batch["output"])
        return batch

    def transform_input(self, x):
        """
        Transform the input.

        @params x: The input
        @return: The transformed input
        """
        return x

    def transform_output(self, y):
        """
        Transform the output.

        @params x: The output
        @return: The transformed output
        """
        return y

    def inv_transform_input(self, x):
        """
        Inverse transform the input.

        @params x: The input
        @return: The inverse transformed input
        """
        return x

    def inv_transform_output(self, y):
        """
        Inverse transform the output.

        @params y: The output
        @return: The inverse transformed output
        """
        return y

class Partial_Standardizer(Data_Transformer):
    """
    A naive standardizer that could be partially fitted to avoid storing a 
        gigantic amount of data. 
    
    @field input_standardizer (sklearn.preprocessing.StandardScaler): The base 
        standardizer for the input part of the dataset.
    @field output_standardizer (sklearn.preprocessing.StandardScaler): The 
        base standardizer for the output part of the dataset.
    """
    def __init__(self, device=device):
        """
        Constructor.
        @params device (torch.device): The device where the parameters are 
            stored. Default is CUDA if CUDA is avaiable and CPU if not.  
        """
        self.super().__init__(device=device)
        self.input_standardizer = sklearn.preprocessing.StandardScaler()
        self.output_standardizer = sklearn.preprocessing.StandardScaler()

    def partial_fit(self, batch: dict):
        """
        Perform a partial fit with a single batch of data. 
        @params batch (dict): The batch of data.
            It must have at least two entries, `"input"` and `"output"`, as 
            the input data and ouput data respectively. All other entries are 
            not modified. 
            CONSTRAINT: "input" in batch and "output" in batch
        """
        x = batch["input"]
        y = batch["output"]
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.input_standardizer.partial_fit(x)
        self.output_standardizer.partial_fit(y)

    def fit(self, dataset, batchsize=None):
        """
        OVERRIDE
        """
        if batchsize is None:
            batchsize = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batchsize)
        for batch in dataloader:
            self.partial_fit(batch)

    def transform_input(self, x):
        """
        OVERRIDE
        """
        x = x.detach().cpu().numpy()
        x = self.input_standardizer.transform(x)
        return torch.tensor(x, device=self.device)

    def transform_output(self, y):
        """
        OVERRIDE
        """
        y = y.detach().cpu().numpy()
        y = self.output_standardizer.transform(y)
        return torch.tensor(y, device=self.device)

    def inv_transform_input(self, x):
        """
        OVERRIDE
        """
        x = x.detach().cpu().numpy()
        x = self.input_standardizer.inverse_transform(x)
        return torch.tensor(x, device=self.device)

    def inv_transform_output(self, y):
        """
        OVERRIDE
        """
        y = y.detach().cpu().numpy()
        y = self.output_standardizer.inverse_transform(y)
        return torch.tensor(y, device=self.device)

class Naive_Standardizer(Data_Transformer):
    """
    A naive standardizer. 

    @field input_mean (torch.Tensor): The mean of the input part of the 
        dataset. 
    @field output_mean (torch.Tensor): The mean of the output part of the 
        dataset. 
    @field input_std (torch.Tensor): The standard deviation of the input part 
        of the dataset. 
    @field output_std (torch.Tensor): The standard deviation of the output part 
        of the dataset. 
    """
    def fit(self, dataset, batchsize=None):
        """
        OVERRIDE
        `batchsize` is not used in this standardizer. 
        """
        dataloader = torch.utils.data.DataLoader(dataset, len(dataset))
        batch = next(iter(dataloader))
        input = batch["input"]
        output = batch["output"]
        self.input_mean = input.mean(0)
        self.output_mean = output.mean(0)
        self.input_std = input.std(0)
        self.output_std = output.std(0)

    def transform_input(self, x):
        """
        OVERRIDE
        """
        return (x - self.input_mean) / self.input_std

    def transform_output(self, y):
        """
        OVERRIDE
        """
        return (y - self.output_mean) / self.output_std

    def inv_transform_input(self, x):
        """
        OVERRIDE
        """
        return x * self.input_std + self.input_mean

    def inv_transform_output(self, y):
        """
        OVERRIDE
        """
        return y * self.output_std + self.output_mean

class PCA(Data_Transformer):
    """
    A principal component analysis (PCA) on the input part of the dataset.
    
    @field model (sklearn.decomposition.PCA): The base PCA model.
    @field use_std (bool): True iff the standard deviation is divided when 
        transforming the data.
    @field device (torch.device): The device where the parameters are stored. 
    """
    def __init__(self, num_dim: int, use_std: bool = True, device=device):
        """
        Constructor.
        @params num_dim (int): The number of dimensions used after reduction. 
        @params use_std (bool, DEFAULT True): True iff the standard deviation 
            is divided when transforming the data.
        @params device (torch.device): The device where the parameters are 
            stored. 
        """
        self.model = sklearn.decomposition.PCA(num_dim)
        self.use_std = use_std
        self.device = device

    def fit(self, dataset, batchsize=None):
        """
        OVERRIDE
        `batchsize` is not used in the data transformer. 
        """
        dataloader = torch.utils.data.DataLoader(dataset, len(dataset))
        batch = next(iter(dataloader))
        input = batch["input"]
        self.input_mean = input.mean(0)
        self.input_std = input.std(0)
        input = input - self.input_mean
        if self.use_std:
            input = input / self.input_std
        self.model.fit(input.detach().cpu().numpy())

    def transform_input(self, x):
        """
        OVERRIDE
        """
        x = x - self.input_mean
        if self.use_std:
            x = x / self.input_std
        x = self.model.transform(x.detach().cpu().numpy())
        x = torch.tensor(x, device=self.device)
        return x

    def inv_transform_input(self, x):
        """
        OVERRIDE
        """
        x = self.model.inverse_transform(x.detach().cpu().numpy())
        x = torch.tensor(x, device=self.device)
        if self.use_std:
            x = x * self.input_std
        x = x + self.input_mean
        return x
    
class Identity_Transformer(Data_Transformer):
    """
    The identity data transformer. 
    """
    def fit(self, dataset, batchsize=None):
        """
        OVERRIDE
        """
        pass

    def transform(self, batch):
        """
        OVERRIDE
        """
        return dict(batch)
    
    def inv_transform(self, batch):
        """
        OVERRIDE
        """
        return dict(batch)


def check_nan(batch: dict):
    """
    Raise assertion error if there is any NAN in the input or the output part 
        of the dataset.
    @params batch (dict): The batch of data.
        It must have at least two entries, `"input"` and `"output"`, as the 
        input data and ouput data respectively. All other entries are not 
        modified. 
        CONSTRAINT: "input" in batch and "output" in batch
        CONSTRAINT: isintance(batch["input"], torch.Tensor)
        CONSTRAINT: isintance(batch["output"], torch.Tensor)
    @error (AssertionError): Error if there is any NAN in the input or the 
        output part of the dataset.
    """
    assert not batch["input"].detach().isnan().any().cpu().item()
    assert not batch["output"].detach().isnan().any().cpu().item()
