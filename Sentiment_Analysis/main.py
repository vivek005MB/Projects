import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import random 
import json
import torchtext
from torchtext.vocab import Vectors, GloVe


from Models.LSTM import LSTM
from Models.bilstm import BILSTM

from utils import LRScheduler, EarlyStopping,accuracy
from train import fit
from load_dataset import load_data



# load the dataset
vocab_size = 25000
path = 'review_data.csv'
with open('config.json','r') as fo:
    config = json.load(fo)
text,data,vocab_size,word_embeddings = load_data(path) #load_dataset.load_data(path)

torch.manual_seed(config['random_seed'])

# Create model
model = BILSTM(vocab_size,config,vec=word_embeddings)

# compile the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
criterion = nn.CrossEntropyLoss()
model.to(device)
optimizer = optim.Adam(model.parameters(),lr=config['learning_rate'])


# fit the model
history = fit(model,data,config,optimizer,criterion,callback=None,device=device)

# plotting
train_loss = history[0]
val_loss = history[2]
train_acc = history[1]
val_acc = history[3]

loss_plot_name = 'train_val_loss'
acc_plot_name = 'train_val_acc'

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"./Plots/{loss_plot_name}.png")
plt.show()

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_acc, color='green', label='train accuracy')
plt.plot(val_acc, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"./Plots/{acc_plot_name}.png")
plt.show()
