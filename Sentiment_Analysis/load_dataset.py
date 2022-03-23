import torch
import torch.nn.functional as F
import torchtext
import time 
import random 
import pandas as pd
import json
from torchtext.vocab import Vectors, GloVe
# torch.backends.cudnn.deterministic = True

# !pip install spacy
# !python -m spacy download en_core_web_sm



def load_data(csv_path): 
    print(f'Loading the data..../')
    with open('config.json','r') as fo:
        config = json.load(fo)
    
    df = pd.read_csv(csv_path)
    random_seed = config['random_seed']
    batch_size = config['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = config['vocab_size']
    
    # text(feature) Processing pipeline
    text = torchtext.legacy.data.Field(
        tokenize = 'spacy',
        tokenizer_language='en_core_web_sm'
    )
    
    # Label processing pipeline
    label = torchtext.legacy.data.LabelField(dtype = torch.long)
    
    # preparing data
    fields = [('final_review',text),('sentiment',label)]
    dataset = torchtext.legacy.data.TabularDataset(
        path = csv_path, format = 'csv',
        skip_header = True,fields = fields
    )   
    
    # dividing train test and validation dataset

    train_data,test_data = dataset.split(
        split_ratio = [0.8,0.2],
        random_state =random.seed(random_seed)
        )
    
    train_data,val_data = train_data.split(
        split_ratio = [0.75,0.25],
        random_state =random.seed(random_seed)
        )
    
    dataset = {'train':train_data,
               'val':val_data,
               'test':test_data}
    print(f"Total training examples = {len(train_data)}")
    print(f"Total validation examples = {len(val_data)}")
    print(f"Total test examples = {len(test_data)}")
    
    # Building Vocab dir(text)
    text.build_vocab(train_data,vectors=GloVe(name='6B',dim=300),max_size = vocab_size)
    label.build_vocab(train_data)
    word_embeddings = text.vocab.vectors
    
    # DataLoader
    train_loader,val_loader,test_loader = torchtext.legacy.data.BucketIterator.splits(
    datasets = (train_data,val_data,test_data),
    batch_size = batch_size,
    device = device,
    sort_within_batch = False,
    sort_key = lambda x: len(x.final_review)
    )
    vocab_size = len(text.vocab)
    print(f'Vocab size : {vocab_size}')
    print(f'Total classes : {len(label.vocab)}')
    dataloader = {'train':train_loader,'val':val_loader,'test':test_loader}
    data = [dataset,dataloader]
    print(f'data loaded successfully..../')
    return text,data,vocab_size,word_embeddings
# '/content/drive/MyDrive/datasets/imdb/review_data.csv'
