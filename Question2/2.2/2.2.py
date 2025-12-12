#!/usr/bin/env python

# Deep Learning Homework 1
# Done by Gonçalo Santos

import argparse
from cProfile import label

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # adicionar o path hm1/DeepLearningHomework1 para importar o modulo utils
import utils

import pandas as pd
class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()
        self.n_classes = n_classes          # 26 para este dataset
        self.n_features = n_features        # 784
        self.hidden_size = hidden_size
        self.layers = layers                # single layer para este exercicio
        self.activation_type = activation_type
        self.dropout = dropout

        # definir a função de ativação dada no argumento
        if self.activation_type == "relu":
            self.activation = nn.ReLU()
        elif self.activation_type == "tanh":
            self.activation = nn.Tanh()

        layers_list = []
        # create the network
        for i in range(self.layers):
            if i == 0:
                # first layer
                layers_list.append(nn.Linear(self.n_features, self.hidden_size))
            else:
                # hidden layers
                layers_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            
            # activation function
            layers_list.append(self.activation)

            if self.dropout > 0:
                layers_list.append(nn.Dropout(self.dropout))
        
        # last layer
        layers_list.append(nn.Linear(self.hidden_size, self.n_classes))
        # put all the layers together
        self.network = nn.Sequential(*layers_list)

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        # passar o input pela rede e obter o output predicted (y_hat)
        scores = self.network(x)
        return scores
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    optimizer.zero_grad()       # sets the gradients to zero
    yhat = predict(model, X)    # compute the model output
    loss = criterion(yhat, y)   # compute the loss
    loss.backward()             # backpropagation (computes the gradients)
    # update model weights using the gradients
    optimizer.step()
    return loss


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    preds = model(X) 
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    # compute the model output
    y_hat = predict(model, X)
    # compute the loss
    loss = criterion(y_hat, y) 
    # compute the accuracy
    accuracy = torch.mean((torch.argmax(y_hat, dim=1) == y).float())  # argmax devolve o indice da classe predicted com maior probabilidade
    return loss, accuracy


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='../../emnist-letters.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    # usar cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # mover tensores para GPU
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    dev_X   = dev_X.to(device)
    dev_y   = dev_y.to(device)
    test_X  = test_X.to(device)
    test_y  = test_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # define the grid search parameters
    grid_search_params = {
        'batch_size': [64],
        'epochs': [30],
        'hidden_size': [16, 32, 64, 128, 256],  # number of units in the hidden layer
        'layers': [1],                          # 1 hidden layer
        'activation': ['relu'],                 # relu or tanh
        'dropout': [0.0, 0.2],                  # No dropout and one non-zero dropout value
        'optimizer': ['adam'],                  # Adam or sgd
        'learning_rate': [0.05, 0.01, 0.005, 0.001], # 4 learning rate values
        'l2_decay': [0.0, 0.01],                # No weight decay (l2 penalty) and one non-zero value
    }

    # para o 2.2(a)
    summary_table = []
    # para o 2.2(b)
    best_val_acc_global = 0.0
    best_model_global_train_losses = []
    best_model_global_train_accs = []
    best_model_global_val_losses = []
    best_model_global_val_accs = []
    # para o 2.2(c)
    best_configs_by_width = {}

    start = time.time()
    for width in grid_search_params['hidden_size']:
        # para o 2.2(c)
        best_val_acc_for_width = 0.0       # reinicia a cada width 

        for lr in grid_search_params['learning_rate']:
            for dropout in grid_search_params['dropout']:
                for l2_decay in grid_search_params['l2_decay']:
                    
                    print(f"Training model with width={width}, lr={lr}, dropout={dropout}, l2={l2_decay}")
                    # re-initialize the model for each configuration
                    model = FeedforwardNetwork(
                        n_classes,
                        n_feats,
                        width,
                        grid_search_params['layers'][0],
                        grid_search_params['activation'][0],
                        dropout
                    ).to(device) # mover o modelo para a GPU
                    
                    # set the loss criterion
                    criterion = nn.CrossEntropyLoss()
                    # set the Adam optimizer
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
                    
                    # train the model
                    train_losses = []
                    train_accs = []
                    valid_losses = []
                    valid_accs = []

                    for epoch in range(opt.epochs):
                        model.train()
                        for batch_X, batch_y in train_dataloader:
                            batch_X = batch_X.to(device)
                            batch_y = batch_y.to(device)
                            loss = train_batch(batch_X, batch_y, model, optimizer, criterion)

                        model.eval()
                        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
                        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)
                        
                        # guardar as losses e accuracies por epoch
                        train_losses.append(train_loss.item())
                        valid_losses.append(val_loss.item())
                        train_accs.append(train_acc.item())
                        valid_accs.append(val_acc.item())

                        
                    
                    # 2.2(a) table summarizing the explored configurations, including
                    # the best validation accuracy obtained during training for each configuration
                    best_val_acc = max(valid_accs)
                    print(f"Best Validation Accuracy: {best_val_acc:.3f}\n")
                    
                    summary_table.append({
                        'width': width,
                        'lr': lr,
                        'dropout': dropout,
                        'l2': l2_decay,
                        'best_val_acc': best_val_acc
                    })

                    # atualizar best config do width atual
                    if best_val_acc > best_val_acc_for_width:
                        best_val_acc_for_width = best_val_acc

                        best_configs_by_width[width] = {
                            "width": width,
                            "lr": lr,
                            "dropout": dropout,
                            "l2": l2_decay,
                            "val_acc": best_val_acc
                        }

                        # salvar o melhor modelo do width atual para o 2.2(c)
                        torch.save(model.state_dict(), f"best_model_width{width}.pt")
                    
                    # guardar o melhor modelo global para o 2.2(b)
                    if best_val_acc > best_val_acc_global:
                        best_val_acc_global = best_val_acc

                        best_global_config = {
                            "width": width,
                            "lr": lr,
                            "dropout": dropout,
                            "l2": l2_decay
                        }

                        # salvar o melhor modelo global atual para o 2.2(b)
                        torch.save(model.state_dict(), "best_model_global.pt")
                        best_model_global_train_losses = train_losses.copy()
                        best_model_global_train_accs = train_accs.copy()
                        best_model_global_val_losses = valid_losses.copy()
                        best_model_global_val_accs = valid_accs.copy()

        # best-performing configuration for each width (according to validation accuracy)
        print(f"\nBest Configuration for width={width}: {best_configs_by_width[width]}\n")
        print(f"Best Validation Accuracy for width={width}: {best_val_acc_for_width:.3f}\n")
    
    # report time taken for training all models
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Best Global Configuration:\n", best_global_config)
    print(f"Best Global Validation Accuracy: {best_val_acc_global:.3f}\n")

    # convert and save summary table to a csv for 2.2(a)
    df = pd.DataFrame(summary_table, columns=["width", "lr", "dropout", "l2", "best_val_acc"])
    df.to_csv("2.2a_gridsearch_results.csv", index=False)

    # 2.2(b) Plot the training and validation loss and the training and validation accuracy over epochs of the best model
    model.load_state_dict(torch.load("best_model_global.pt", weights_only=True)) # carregar o melhor modelo global
    config_best_model = (
        f"width-{best_global_config['width']}-lr-{best_global_config['lr']}"
        f"-dropout-{best_global_config['dropout']}-l2-{best_global_config['l2']}"
    )

    losses = {
        'Train Loss': best_model_global_train_losses,
        'Validation Loss': best_model_global_val_losses
    }

    accuracies = {
        'Train Accuracy': best_model_global_train_accs,
        'Validation Accuracy': best_model_global_val_accs
    }
    
    # plot
    epochs = list(range(1, opt.epochs + 1))
    plot(epochs, losses, filename=f'2.2b_best_model_loss-{config_best_model}.pdf')
    plot(epochs, accuracies, filename=f'2.2b_best_model_accuracy-{config_best_model}.pdf')
    
    # Report the test accuracy of the best model
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss, test_acc = evaluate(model, test_X, test_y, criterion)
    print(f"Best Model Test Accuracy : {test_acc.item():.3f}")

    # 2.2(c) Using the best-performing model for each hidden-layer width,
    # analyze how model width affects the network’s capacity to interpolate the training data.
    # interpolação dos dados de treino refere-se à capacidade do modelo de memorizar os dados de treino (accuracy no treino)
    final_train_accuracies = []
    widths = grid_search_params['hidden_size']

    for width in widths:
        cfg = best_configs_by_width[width]
        # re-initialize the model for each width
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            width,
            grid_search_params['layers'][0],
            grid_search_params['activation'][0],
            dropout=cfg["dropout"]                       
        ).to(device)                            # mover o modelo para a GPU

        # carregar o melhor modelo para cada width
        model.load_state_dict(torch.load(f"best_model_width{width}.pt", weights_only=True))
        # avaliar a accuracy no training set
        model.eval()
        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
        final_train_accuracies.append(train_acc.item())
        print(f"Width: {width}, Final Training Accuracy: {train_acc.item():.3f}")

    # plot the final training accuracy as a function of hidden-layer width
    plt.clf()
    plt.xlabel('Hidden Layer Width')
    plt.ylabel('Final Training Accuracy')
    plt.plot(widths, final_train_accuracies, label='Final Training Accuracy')
    plt.legend()
    plt.savefig('2.2c_final_training_acc_vs_width.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()