import pandas as pd
import glob
from matplotlib import pyplot as plt
from torch import nn
import torch
import numpy as np
import math
import time
from torch.utils.data import DataLoader
from os.path import exists



def test_train_split(data : torch._tensor, classes : torch.tensor, random_seed : int, p_train : float):
    indices = list(range(len(data)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_index = math.ceil(len(data) * (1-p_train))
    i_train = indices[:split_index]
    i_test = indices[split_index:]
    train_data = data[i_train]
    train_labels = classes[i_train]
    test_data = data[i_test]
    test_labels = classes[i_test]
    return train_data, train_labels, test_data, test_labels

def run_test_set(test_data : torch.tensor, test_labels : torch.tensor , model):
    """ 
    runs the dataset through the model and returns the proportion correctly
    classified, and the percentage for which the correct answer was in the 
    top two outputs
    """
    model.eval()
    pred = model(test_data)
    n = pred.shape[0]
    count_correct = 0
    count_top_two = 0
    for i in range(n):
        top_values, top_indices = torch.topk(pred[i],2)
        label = torch.argmax(test_labels[i])
        count_correct += int(top_indices[0] == label)  
        count_top_two += int(label in top_indices)
    model.train()
    return count_correct/n, count_top_two/n

class Model(nn.Module):
    """ defining a simple classifier network, outputting softmax for probabilities"""
    def __init__(self, l_in : int, l_mid : int, l_out : int):
        super(Model,self).__init__()
        self.network_seq = nn.Sequential(
        nn.Linear(l_in,l_mid),
        nn.Sigmoid(),
        nn.Linear(l_mid, l_out),
        nn.Softmax(dim=-1)
        )

    def forward(self, x):
      out = self.network_seq(x)
      return out


def load_data(data_file : str, label_file : str, device : str) -> torch.tensor:
    """ 
    this loads the data and labels into tensors 
    note that the data is cast to float32 before being converted to a tensor
    """
    # TODO: create transformations of the input data to increase the amount 
    data = pd.read_csv(data_file)
    labels = pd.read_csv(label_file)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    labels.drop(columns=labels.columns[0], axis=1, inplace=True)
    data = torch.tensor(np.float32(data.to_numpy())).to(device)
    labels = torch.tensor(np.float32(labels.to_numpy())).to(device)

    return data, labels

def backprop_epoch(model, dataloader, optimiser, loss_function, losses : list[float]):
    epochStart = time.time()
    for i, data in enumerate(dataloader, 0):
        optimiser.zero_grad()
        yPrediction = model(data[0])
        #print(yPrediction[0],data[1][0])
        loss = loss_function(yPrediction, data[1])
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    return model, optimiser, losses

def train_model(load_model : bool, model_path : str, epochs : int, model, dataloader, optimiser, loss_function, test_data, test_labels, losses, test_losses):

    test_correct = []
    test_top_two = []
    for i in range(epochs):
        model, optimiser, losses = backprop_epoch(model, dataloader, optimiser, loss_function, losses)
        a,b = run_test_set(test_data, test_labels, model)
        test_correct.append(a)
        test_top_two.append(b)
        test_losses.append(loss_function(model(test_data),test_labels).item())
        if (i+1) % 20 == 0:
            print('Epoch [{}/{}], loss of {:.3f}, and the percentage correctly classified is {:.1f}%'.format(i+1,epochs,losses[-1],100*test_correct[-1]))
            

    return model, optimiser, test_correct, test_top_two, losses, test_losses

def main():
    
    data_file = "data.csv"
    label_file = "labels.csv"
    model_path = "train_semaphore.pt"

    TEST_TRAIN_PROPORTION = 0.25
    SEED = 999
    BATCH_SIZE = 16
    lr = 0.001
    epochs = 1000
    save_model = True
    load_model = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    data, labels = load_data(data_file, label_file, device)
    print("The data has shape {}, and the labels have shape {}".format(data.shape, labels.shape))

    train_data, train_labels, test_data, test_labels = test_train_split(data, labels, SEED, TEST_TRAIN_PROPORTION)

    model = Model(38, 60, 27)
    print("The initialised model is correct {:.1f}% of the time, against an expected {:.1f}%".format(run_test_set(data, labels, model)[0]*100, 100*1/27.0))

    optimiser = torch.optim.Adam(model.parameters(), lr = lr)
    dataloader = DataLoader(list(zip(train_data, train_labels)), batch_size=BATCH_SIZE, shuffle=True)
    loss_function = nn.BCELoss()

    if load_model and exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['loss']
        print('Loaded model at ' + model_path)
    else:
        losses = []
    test_losses = []

    model, optimiser, test_correct, test_top_two, losses, test_losses = train_model(load_model, model_path, epochs, model, dataloader, optimiser, loss_function, test_data, test_labels, losses, test_losses)

    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss': losses[-1],
                }, model_path)
        print('Model saved as "' + model_path + '"')


    plt.figure(0)
    l = len(losses)
    plt.plot([i*epochs/l for i in range(l)],losses, label="train")
    l2 = len(test_losses)
    plt.plot([i*epochs/l2 for i in range(l2)],test_losses, label="test")
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('loss')
    plt.show()

    plt.figure(1)
    l = len(test_correct)
    plt.plot([i*epochs/l for i in range(l)],test_correct, label="correct")
    l2 = len(test_top_two)
    plt.plot([i*epochs/l2 for i in range(l2)],test_top_two, label="top two")
    plt.xlabel('epochs')
    plt.legend()
    plt.ylabel('proportion correct')
    plt.show()

if __name__ == "__main__":
    main()