import torch 
import torch.nn as nn
import torch.nn.functional as F
#config is dictionary(can be JSON file also) that contains  hyperparameters
class BILSTM(nn.Module):
    """
    Arguments:
    ___
    vocab_size : size of vocabulary (unique words)
    config : dictionary containing  embedding_dimension,hidden_dimension
    vec : Pre-trained word embedding (GloVe embedding)
    ___

    """
    def __init__(self,vocab_size,config,vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.embedding_dim =  config['embedding_dim']
        self.gpu = config['gpu'] # boolean [True/False]
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim=config['embedding_dim'])
        
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #loading pretrained word embeddings e.g GloVe
            # self.embedding.weights = nn.Parameter(weights,requires_grad=False)
            #self.embedding.weight.requires_grad = False #non-trainable
        self.lstm = nn.LSTM(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_dim'],
            num_layers = config['n_layers'],
            bidirectional = config['is_bidirectional'],
            # dropout = config['dropout']
            )
        self.fc = nn.Linear(in_features = config['hidden_dim'],out_features=config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def attention(self,lstm_output,final_hidden):
        # lstm_output shape                      [batch_size,sequence_length, hidden_dim]
        # final_hidden state shape               [1,batch_size,hidden_dim]     
        
        hidden = final_hidden.squeeze(0)   
        # hidden shape :                         [batch_size,hidden_dim]  
        
        attention_weights = torch.bmm(lstm_output,hidden.unsqueeze(2)).squeeze(2)
        # attention_weights shape :              [batch_size, sequence_length]  
        
        soft_attention_weights = F.softmax(attention_weights,1)
        # soft attention weights :               [batch_size,sequence_length]
        
        new_hidden = torch.bmm(lstm_output.transpose(1,2),soft_attention_weights.unsqueeze(2)).squeeze(2)
        # new hidden shape :                     [batch_size,hidden_dim]
        return new_hidden
    
    def forward(self,text):
        # text shape :         [sequence_length, batch_size]
        
        embedded = self.embedding(text)
        embedded = self.dropout(embedded) #dropout 
        # embedded shape :     [sequence_length,batch_size,embedding_dim]

        output, (hidden,cell) = self.lstm(embedded)
        # output shape :       [sequence_length, batch_size, 2* hidden_dim] 2 because of bi-directional
        # hidden state shape : [2*num_layers,batch_size,hidden_dim] 
        # cell state shape :   [2*num_layers,batch_size,hidden_dim] 
        
        fbout = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:] # sum of bidirectional output F+B
        # fbout shape :        [sequence_length, batch_size, hidden_dim]
        
        fbout = fbout.permute(1,0,2)
        # fbout shape :        [batch_size,sequence_length, hidden_dim]

        fb_hidden = (hidden[-2,:,:]+hidden[-1,:,:]).unsqueeze(0)
        # fb_hidden shape :    [1,batch_size,hidden_dim]
        
        attention_output = self.attention(fbout,fb_hidden)
        # attention output :   [batch_size,hidden_dim]

        logits = self.fc(attention_output)
        # logits shape :       [batch_size,out_dim]
        return logits