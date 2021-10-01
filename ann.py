import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class RNN(nn.Module):      
    
    def __init__(self, input_size):
        super(RNN, self).__init__()

        self.num_layers = 3
        self.hidden_size = 128

        self.encoder = nn.Embedding(input_size, self.hidden_size, padding_idx=0)

        self.lstm = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=False)
        
        self.decoder = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Linear(self.hidden_size, input_size)
            #nn.Softmax(dim=2)
        )

        self.cuda()
    
    def forward(self, x, h):

        x = self.encoder(x)
        x, h = self.lstm(x, h)

        #print(x.shape)
        out = self.decoder(x)

        return out, h
    
    def init_state(self, sequence_length=32, cuda=True):
        # batch_size x hidden_size
        if cuda:
            return (torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda(),
                    torch.zeros(self.num_layers, sequence_length, self.hidden_size).cuda())

        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))










