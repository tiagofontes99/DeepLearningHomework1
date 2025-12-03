#!/usr/bin/env python

# Deep Learning Homework 1
# Feito por Tiago Fontes

import argparse
import time
import pickle
import json
import numpy as np
import utils
import extractHOGFeatures
import os

DATA_PATH = r"C:\Users\CeX\PycharmProjects\DeepLearningHomework1\Question1\emnist-letters.npz"

'''Para logistic Regression multiclass usmos a softmax 
a formula é p(y = k|x) = e^wk.T@x/ sumj(e^wj.T@x)'''


class LogisticRegression:
    # Inicialização dos pesos a 0 onde se vai fazer o update
    # array com formato n_classes * n*features = 26 * 784
    def __init__(self, n_classes, n_features, eta, l2pen):
        self.W = np.zeros((n_classes, n_features))
        self.eta = eta
        self.l2pen = l2pen


    def evaluate(self, X, y):
        """
        X examples whith 768 features
        returns score in float
        """
        y_true = y - 1
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y_true)
        return acc

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_weight(self, x_i, y_i):
        """
        x_i (784,): a single training example
        y_i (numero de 1 a 26): the gold label for that example

        """
        y_idx = y_i - 1

        scores = self.W @ x_i
        scores = scores - np.max(scores)
        exps = np.exp(scores)
        probs = exps / np.sum(exps)

        y_onehot = np.zeros_like(probs)
        y_onehot[y_idx] = 1.0

        grad_W = np.outer(probs - y_onehot, x_i)  # (C, F)

        self.W -= self.eta * (grad_W + self.l2pen * self.W)
        pass

    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        for i in range(X.shape[0]):
            self.update_weight(X[i], y[i])

        pass

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        scores = X @ self.W.T
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exps = np.exp(scores)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        return np.argmax(probs, axis=-1)

    def get_args(self):
        return {"eta": self.eta, "pen": self.l2pen, "HOG features": self.hog}


def main(args):
    utils.configure_seed(seed=args.seed)
    print(args)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]
    eta_value = [0.01, 0.001, 0.0001]
    l2pen = [0.0001, 0.00001]
    # initialize the model

    best_global = 0.0
    best_l2 =0.0
    best_eta = 0.0
    hog = False
    for i in range(2):
        if i == 1:
            X_train = extractHOGFeatures.extract_hog_features(X_train)
            X_valid = extractHOGFeatures.extract_hog_features(X_valid)
            X_test = extractHOGFeatures.extract_hog_features(X_test)
            hog= True
        for eta in eta_value:
            for l2 in l2pen:
                best_valid = 0.0
                best_model = False

                model = LogisticRegression(n_classes, X_train.shape[1], eta, l2)

                epochs = np.arange(1, args.epochs + 1)

                valid_accs = []
                train_accs = []

                start = time.time()


                best_epoch = -1
                for i in epochs:
                    print('Training epoch {}'.format(i))
                    train_order = np.random.permutation(X_train.shape[0])
                    X_train = X_train[train_order]
                    y_train = y_train[train_order]

                    model.train_epoch(X_train, y_train)

                    train_acc = model.evaluate(X_train, y_train)
                    valid_acc = model.evaluate(X_valid, y_valid)

                    train_accs.append(train_acc)
                    valid_accs.append(valid_acc)

                    print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

                    if valid_acc > best_valid:
                        best_valid = valid_acc
                        best_epoch = i
                        print(f"New best model at epoch {i} with val acc = {valid_acc:.4f}")

                    if best_valid > best_global:
                        best_global = best_valid
                        model.save(args.save_path)
                        best_eta=eta
                        best_l2 = l2
                        best_model=True
                        model.save(args.save_path)
                        print(f"New best global with param = l2: {l2}, eta {eta}, hog { hog}, val acc = {valid_acc:.4f}")
                        with open(args.best_score, "w") as f:
                            json.dump({
                            "best_global": float(best_global),
                            "l2": best_l2,
                            "eta": best_eta,
                            "HOG features": hog,
                        }, f, indent=4)

                elapsed_time = time.time() - start
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                print('Training took {} minutes and {} seconds'.format(minutes, seconds))

                print("Reloading best checkpoint")
                best_model = LogisticRegression.load(args.save_path)
                test_acc = best_model.evaluate(X_test, y_test)

                print('Best model test acc: {:.4f}'.format(test_acc))
                if best_model:
                    utils.plot(
                        "Epoch", "Accuracy",
                        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
                        filename=args.accuracy_plot
                    )
                results = {
                    "best_valid": float(best_valid),
                    "test": float(test_acc),
                    "time": elapsed_time,
                    "l2": l2,
                    "eta": eta,
                    "HOG features": hog,
                }

                if os.path.exists(args.scores):
                    with open(args.scores, "r") as f:
                        data = json.load(f)

                    if isinstance(data, dict):
                        data = [data]

                else:
                    data = []
                data.append(results)
                with open(args.scores, "w") as f:
                    json.dump(data, f, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="logistic-grid_search.npz")
    parser.add_argument("--accuracy-plot", default="Q1-logistic-accs-grid_search.pdf")
    parser.add_argument("--scores", default="Q1-logistic-scores-grid_search.json")
    parser.add_argument("--best_score", default="Q1-logistic-scores-grid_search_best_param.json")
    args = parser.parse_args()
    main(args)
