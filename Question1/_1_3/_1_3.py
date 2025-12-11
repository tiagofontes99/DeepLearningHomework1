#!/usr/bin/env python

# Deep Learning Homework 1
#Feito por Tiago Fontes

import argparse
import time
import pickle
import json
import numpy as np

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "emnist-letters.npz")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # para poder dar import utils
import utils

class MultiLayerPerceptron:
    def __init__(self, n_classes, n_features, hidden_dim=100, eta=0.001):
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.eta = eta

        # Inicialização
        # W ~ N(µ, σ^2) com µ = 0.1, σ = 0.1
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_dim, n_features))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_dim))
        self.b2 = np.zeros(n_classes)

    # ReLU e derivada
    def relu(self, z):
        return np.maximum(0, z)

    def drelu(self, z):
        return (z > 0).astype(float)

    def forward(self, x):
        """
        x: (n_features,)
        devolve z1, h, z2
        """
        z1 = self.W1 @ x + self.b1
        h  = self.relu(z1)
        z2 = self.W2 @ h + self.b2
        return z1, h, z2

    def predict(self, X):
        """
        X: (n_examples, n_features) ou (n_features,)
        devolve labels em [0..n_classes-1]
        """
        if X.ndim == 1:
            _, _, z2 = self.forward(X)
            return np.argmax(z2)
        else:
            scores = np.empty((X.shape[0], self.n_classes))
            for i in range(X.shape[0]):
                _, _, z2 = self.forward(X[i])
                scores[i] = z2
            return np.argmax(scores, axis=1)

    def update_weight(self, x_i, y_i):
        """
        x_i: (n_features,)
        y_i: label em [1..n_classes]
        devolve o loss da amostra (para logging)
        """
        y_idx = y_i - 1  # converter para 0-based

        # Forward
        z1, h, z2 = self.forward(x_i)

        # Softmax
        z2_shift = z2 - np.max(z2)
        exp_scores = np.exp(z2_shift)
        probs = exp_scores / np.sum(exp_scores)

        # Cross-entropy
        loss = -np.log(probs[y_idx] + 1e-12)

        # Backprop
        # grad_z2 = p - y_onehot
        grad_z2 = probs.copy()
        grad_z2[y_idx] -= 1.0

        # Gradientes output layer
        grad_W2 = np.outer(grad_z2, h)
        grad_b2 = grad_z2

        # Backprop para hidden
        grad_h  = self.W2.T @ grad_z2
        grad_z1 = grad_h * self.drelu(z1)

        grad_W1 = np.outer(grad_z1, x_i)
        grad_b1 = grad_z1

        # SGD update
        self.W2 -= self.eta * grad_W2
        self.b2 -= self.eta * grad_b2
        self.W1 -= self.eta * grad_W1
        self.b1 -= self.eta * grad_b1

        return loss

    def train_epoch(self, X, y):
        """
        SGD com batch size = 1
        devolve loss médio do epoch
        """
        total_loss = 0.0
        for i in range(X.shape[0]):
            total_loss += self.update_weight(X[i], y[i])
        return total_loss / X.shape[0]

    def evaluate(self, X, y):
        """
        accuracy; y vem em [1..n_classes]
        """
        y_true = y - 1
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y_true)
        return acc

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]


    model = MultiLayerPerceptron(n_classes, n_feats, hidden_dim=100, eta=0.001)

    epochs = np.arange(1, args.epochs + 1)

    train_accs = []
    valid_accs = []
    train_losses = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1

    for i in epochs:
        print(f'Training epoch {i}')

        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        avg_loss = model.train_epoch(X_train, y_train)
        train_losses.append(avg_loss)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print(f'train loss: {avg_loss:.4f} | train acc: {train_acc:.4f} | val acc: {valid_acc:.4f}')

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            model.save(args.save_path)
            print(f"New best model at epoch {i} with val acc = {valid_acc:.4f}")

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f'Training took {minutes} minutes and {seconds} seconds')

    print("Reloading best checkpoint")
    best_model = MultiLayerPerceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print(f'Best model test acc: {test_acc:.4f}')

    # Plot das accuracies
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    # Plot do train loss
    utils.plot(
        "Epoch", "Train loss",
        {"train_loss": (epochs, train_losses)},
        filename="Q3-mlp-loss.pdf"
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="mlp.npz")
    parser.add_argument("--accuracy-plot", default="Q3-mlp-accs.pdf")
    parser.add_argument("--scores", default="Q3-mlp-scores.json")
    args = parser.parse_args()
    main(args)
