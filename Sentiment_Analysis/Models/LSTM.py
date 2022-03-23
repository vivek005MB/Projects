
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    """
    Arguments:
    ___
    vocab_size : size of vocabulary (unique words)
    config : dictionary containing  embedding_dimension,hidden_dimension
    vec : Pre-trained word embedding (GloVe embedding)
    ___

    """
    def __init__(self,vocab_size, config,vec=None):
        super().__init__()      

        self.embedding = nn.Embedding(vocab_size,config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'],config['hidden_dim'])
        self.fc = nn.Linear(config['hidden_dim'],config['out_dim'])
        self.dropout  = nn.Dropout(config['dropout'])
        
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #loading pretrained word embeddings e.g GloVe
    
    def forward(self,text):
        # text shape : [sequence_length, batch_size]

        embedded = self.dropout(self.embedding(text))
        # embedded shape : [sequence_length,batch_size,embedding_dim]

        output,(hidden,cell) = self.lstm(embedded)
        # output shape : [sequence_length,batch_size,hidden_dim]
        # hidden shape : [1,batch_size,hidden_dim]
        
        hidden.squeeze_(0) 
        # hidden shape : [batch_size, hidden_dim]

        output = self.fc(hidden)
        # ouput shape : [batch_size,output_dim]

        return output