#!/usr/bin/env python

import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation_type = activation_type
        self.dropout = dropout

        if self.activation_type == "relu":
            self.activation = nn.ReLU()
        elif self.activation_type == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unknown activation")

        layers_list = []
        for i in range(self.layers):
            if i == 0:
                layers_list.append(nn.Linear(self.n_features, self.hidden_size))
            else:
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers_list.append(self.activation)
            if self.dropout > 0:
                layers_list.append(nn.Dropout(self.dropout))

        layers_list.append(nn.Linear(self.hidden_size, self.n_classes))
        self.network = nn.Sequential(*layers_list)

    def forward(self, x, **kwargs):
        scores = self.network(x)
        return scores


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(model, X, y, criterion):
    y_hat = model(X)
    loss = criterion(y_hat, y)
    acc = torch.mean((torch.argmax(y_hat, dim=1) == y).float())
    return loss, acc


def plot(epochs, plottables, filename=None, ylim=None, ylabel=None):
    plt.clf()
    plt.xlabel('Epoch')
    if ylabel is not None:
        plt.ylabel(ylabel)
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def main():
    epochs = 30
    batch_size = 64
    hidden_size = 32
    activation = "relu"
    dropout = 0.0
    lr = 0.001
    l2_decay = 0.0

    utils.configure_seed(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_path = "emnist-letters.npz"
    data = utils.load_dataset(data_path)
    dataset = utils.ClassificationDataset(data)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)
    test_X, test_y = dataset.test_X.to(device), dataset.test_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    depth_list = [1, 3, 5, 7, 9]

    summary_depth = []
    best_val_global = 0.0
    best_global_config = None
    best_global_train_losses = []
    best_global_train_accs = []
    best_global_val_losses = []
    best_global_val_accs = []
    final_train_acc_by_depth = {}

    criterion = nn.CrossEntropyLoss().to(device)

    start = time.time()

    for depth in depth_list:
        print(f"\n=== Training model with depth={depth} ===")

        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            hidden_size,
            depth,
            activation,
            dropout
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2_decay
        )

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            model.train()
            epoch_losses = []

            for X_batch, y_batch in train_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
                epoch_losses.append(loss.item())

            model.eval()
            train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
            val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

            train_losses.append(train_loss.item())
            train_accs.append(train_acc.item())
            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())

            print(
                f"Epoch {epoch + 1:02d} | "
                f"train_loss={train_loss.item():.4f} "
                f"| val_loss={val_loss.item():.4f} "
                f"| val_acc={val_acc.item():.4f}"
            )

        best_val_acc = max(val_accs)
        summary_depth.append({
            "depth": depth,
            "best_val_acc": best_val_acc
        })
        final_train_acc_by_depth[depth] = train_accs[-1]

        print(f"Best Validation Accuracy for depth={depth}: {best_val_acc:.3f}")

        if best_val_acc > best_val_global:
            best_val_global = best_val_acc
            best_global_config = {
                "depth": depth,
                "hidden_size": hidden_size,
                "lr": lr,
                "dropout": dropout,
                "l2": l2_decay
            }
            best_global_train_losses = train_losses.copy()
            best_global_train_accs = train_accs.copy()
            best_global_val_losses = val_losses.copy()
            best_global_val_accs = val_accs.copy()
            torch.save(model.state_dict(), "best_model_depth_global.pt")

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('\nTraining took {} minutes and {} seconds'.format(minutes, seconds))

    print("\nSummary (depth, best_val_acc):")
    for row in summary_depth:
        print(f"Depth={row['depth']}, Best Val Acc={row['best_val_acc']:.3f}")

    print("\nBest Global Configuration:", best_global_config)
    print(f"Best Global Validation Accuracy: {best_val_global:.3f}")

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        best_global_config["depth"],
        activation,
        dropout
    ).to(device)
    model.load_state_dict(torch.load("best_model_depth_global.pt", map_location=device))
    model.eval()
    test_loss, test_acc = evaluate(model, test_X, test_y, criterion)
    print(f"\nBest Depth Test Accuracy: {test_acc.item():.3f}")

    # 2.3(b):
    config_best = (
        f"depth-{best_global_config['depth']}-width-{hidden_size}-"
        f"lr-{lr}-dropout-{dropout}-l2-{l2_decay}-opt-adam"
    )
    epochs_axis = list(range(1, epochs + 1))

    losses = {
        "Train Loss": best_global_train_losses,
        "Valid Loss": best_global_val_losses
    }
    accuracies = {
        "Train Accuracy": best_global_train_accs,
        "Valid Accuracy": best_global_val_accs
    }

    plot(epochs_axis, losses,
         filename=f"2.3b_best_depth_loss-{config_best}.pdf",
         ylabel="Loss")
    plot(epochs_axis, accuracies,
         filename=f"2.3b_best_depth_accuracy-{config_best}.pdf",
         ylabel="Accuracy")

    # 2.3(c):
    depths_sorted = sorted(final_train_acc_by_depth.keys())
    final_train_accs = [final_train_acc_by_depth[d] for d in depths_sorted]

    plt.clf()
    plt.xlabel('Depth (number of hidden layers)')
    plt.ylabel('Final Training Accuracy')
    plt.plot(depths_sorted, final_train_accs, marker='o', label='Final Training Accuracy')
    plt.legend()
    plt.savefig('2.3c_final_training_acc_vs_depth.pdf', bbox_inches='tight')

    print("\nFinal training accuracies by depth:")
    for d in depths_sorted:
        print(f"Depth={d}, Final Train Acc={final_train_acc_by_depth[d]:.3f}")


main()
