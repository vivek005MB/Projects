import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
from Models.LSTM import LSTM
from Models.bilstm import BILSTM
from utils import LRScheduler, EarlyStopping,accuracy
from tqdm import tqdm
import pandas as pd
import torchtext
from torchtext.vocab import Vectors, GloVe
import random 
import pandas as pd
import json

# import load_dataset
vocab_size = 25000




def fit(model,data,config,optimizer,criterion,callback=None,device='cpu'):
    """
    Arguments:
    _________
    model    : instance of LSTM/BiLSTM class
    data     : list containing dataset dictionary and dataloader dictionary (train,val,test)
    config   : configuration json file containing hyperparameters
    callback : None,'lr_scheduler','early_stopping' (select one, default None)

    """
    start_time = time.time()
    train_loss,train_accuracy = [],[]
    val_loss,val_accuracy = [], []
    dataloader = data[1]
    dataset = data[0]
    if callback == 'lr_scheduler':
        lr_scheduler = LRScheduler(optimizer)
    if callback == 'early_stopping':
        early_stopping = EarlyStopping()
    
    for epoch in range(config['num_epochs']):
        model.train()
        
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        total = 0
        
        # progress_bar = tqdm(enumerate(dataloader['train']), total =int(len(dataset['train'])/dataloader['train'].batch_size))
        # for batch_idx, batch in progress_bar:
        for batch_idx,batch in enumerate(dataloader['train']):
            counter += 1
            
            text = batch.final_review.to(device)
            label = batch.sentiment.to(device)

            optimizer.zero_grad()

            # forward pass
            logits = model(text)
            loss = criterion(logits,label)
            
            train_running_loss += loss.item() 
            _,preds = torch.max(logits,dim=1)
            train_running_correct += (preds == label).sum().item()
            total += label.size(0)

            # backward pass
            loss.backward()
            
            # update model parameters
            optimizer.step()
            
            # Booking Keeping
            if batch_idx % 50 == 0:
                print(f" Epoch :{epoch+1:03d}/{config['num_epochs']:03d} | Batch Number = {batch_idx+1:03d}/{len(dataloader['train'])} | Loss : {loss:.4f}")
        
        # with torch.set_grad_enabled(False):
        #     print(f"training Accuracy : {accuracy(model,dataloader['train'],device):.2f}")
        #     print(f"Validation Accuracy : {accuracy(model,dataloader['val'],device):.2f}")
        
        train_loss.append(train_running_loss / counter)
        train_accuracy.append(100. * train_running_correct / total)
        
       
        val_running_loss = 0.0
        val_running_correct = 0
        counter_val = 0
        total_val = 0
        model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(dataloader['val']):
                counter_val += 1
                
                text = batch.final_review.to(device)
                label = batch.sentiment.to(device)   

                
                total_val += label.size(0)
                outputs = model(text)
                loss = criterion(outputs, label)
                
                val_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                val_running_correct += (preds == label).sum().item()
            
            val_loss.append(val_running_loss / counter_val)
            current_val_loss = val_running_loss / counter_val
            val_accuracy.append( 100. * val_running_correct / total_val)
       
        if callback == 'lr_scheduler':
            lr_scheduler(current_val_loss)
        elif callback == 'early_stopping':
            early_stopping(current_val_loss)
            if early_stopping.early_stop:
                break
        print(10*'-----') 
        print(f"After epoch {epoch:02d} | Training Loss = {(train_running_loss / counter):.4f} | Train Accuracy = {(100. * train_running_correct / total):.4f}|")
        print(f"After epoch {epoch:02d} | Validation Loss = {(val_running_loss / counter_val):.4f} | Validation Accuracy = {( 100. * val_running_correct / total_val):.4f}|")
        print(10*'-----')  

    print(f'Total time for training is {(time.time()-start_time)/60:.2f} minutes.')
    print(f"Accuracy on the Test Set is {accuracy(model,dataloader['test'],device):.4f}")
    print('Saving model...')
    torch.save(model.state_dict(), f"./checkpoint.pth")
    return train_loss,train_accuracy,val_loss,val_accuracy