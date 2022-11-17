import os

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_channels = 1
    shape = (28, 28)
    for i in range(n_layers):
        out_channels = trial.suggest_int("n_channels_l{}".format(i), 4, 128)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        layers.append(nn.MaxPool2d(kernel_size=2))
        in_channels = out_channels
        shape = (shape[0] // 2, shape[1] // 2)
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_channels * shape[0] * shape[1], 10))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_dataset():
    cwd = os.getcwd()
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(cwd, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(cwd, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
    )

    return train_loader, valid_loader


def objective(trial):

    # モデル・オプティマイザ・データセットを用意する
    model = define_model(trial)
    optimizer = optim.Adam(model.parameters())
    train_loader, valid_loader = get_dataset()

    # 学習ループ
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_loader.dataset)

        # 中間評価値を報告する (1)
        trial.report(accuracy, epoch)

        # 枝刈りをするべきか判定する (2)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
fig = optuna.visualization.plot_intermediate_values(study)
fig.write_image("ch3_intermediate_values.png")
