import torch
from torch import nn
import torch.optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cpu = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = cpu
if __name__ == "__main__":
    print(f"Currently using device: {device}")

class Naive_Net(nn.Module):
    def __init__(self, layer_sizes):
        assert len(layer_sizes) > 1
        super(Naive_Net, self).__init__()
        layers = [nn.BatchNorm1d(layer_sizes[0])]
        i = -1
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[i+1], layer_sizes[i+2]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Bhattacharyya_Loss(nn.Module):
    def forward(self, x, y):
        x, sx = x[:, 0], x[:, 1]
        y, sy = y[:, 0], y[:, 1]
        sx2, sy2 = sx**2, sy**2
        return ((x - y)**2 / (sx2 + sy2)).sum() + \
            2 * torch.log((sx2 + sy2) / (2 * sx.abs() * sy.abs())).sum()

class Physical_Loss(nn.Module):
    def __init__(self, regularization):
        super(Physical_Loss, self).__init__()
        self.regularization = regularization

    def forward(self, x, y):
        x, sx = x[:, 0], x[:, 1]
        y, sy = y[:, 0], y[:, 1]
        sx2, sy2 = sx**2, sy**2
        return ((x - y)**2 / (sx2 + sy2)).sum() + \
            self.regularization * sx2.sum()

class Naive_Net_Model():
    def __init__(self, dataloader, layer_sizes,
            loss="MSE", regularization=0., device=device):
        self.dataloader = dataloader
        self.regularization = regularization
        if loss in ["Bhattacharyya", "Physical"]:
            layer_sizes[-1] += 1
        self.model = Naive_Net(layer_sizes)
        self.validation = False
        self.device = device
        if loss == "MSE":
            self.loss_fn = nn.MSELoss()
        elif loss == "Physical":
            self.loss_fn = Physical_Loss(self.regularization)
        elif loss == "Bhattacharyya":
            self.loss_fn = Bhattacharyya_Loss()
        else:
            raise ValueError(f"Loss {loss} not implemented!")

    def set_validation(self, val_dataloader):
        self.validation = True
        self.val_dataloader = val_dataloader

    def train(self, num_epochs, lr, if_print=False):
        self.model.train()
        if self.validation:
            self.val_losses = []
            self.val_mse_losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            if if_print:
                print(f"Training epoch {epoch}")
            for i, batch in enumerate(self.dataloader):
                optimizer.zero_grad()
                pred = self.model(batch["input"])
                loss = self.loss_fn(pred, batch["output"])
                loss.backward()
                optimizer.step()
            if if_print:
                print("Training loss", loss.detach().cpu().item())

            if self.validation:
                self.model.eval()
                for batch in self.val_dataloader:
                    val_pred = self.model(batch["input"])
                    val_loss = self.loss_fn(val_pred, batch["output"])
                    val_loss = val_loss.detach().cpu().item()
                    mse_loss = nn.MSELoss()(val_pred[:, 0], batch["output"][:, 0])
                    mse_loss = mse_loss.detach().cpu().item()
                    self.val_losses.append(val_loss)
                    self.val_mse_losses.append(mse_loss)
                    if if_print:
                        print("Validation loss:", val_loss)
                    break
                self.model.train()

    def predict(self, input):
        self.model.eval()
        return self.model(input)
