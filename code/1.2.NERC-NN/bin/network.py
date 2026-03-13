
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

class nercLSTM(nn.Module):
   def __init__(self, codes, hyperparams) :
      super(nercLSTM, self).__init__()
      self.activation = hyperparams.get('activation', 'relu')

      n_lc_words = codes.get_n_lc_words()
      n_words = codes.get_n_words()
      n_sufs = codes.get_n_sufs()
      n_feat = codes.get_n_features()
      n_labels = codes.get_n_labels()

      embLWsize = hyperparams['embLWsize']
      embWsize = hyperparams['embWsize']
      embSsize = hyperparams['embSsize']  
      self.embLW = nn.Embedding(n_lc_words, embLWsize)
      self.embW = nn.Embedding(n_words, embWsize)
      self.embS = nn.Embedding(n_sufs, embSsize)
      
      dropout = hyperparams['dropout']
      self.dropLW = nn.Dropout(dropout)
      self.dropW = nn.Dropout(dropout)
      self.dropS = nn.Dropout(dropout)
      
      lstm_in_size = embLWsize + embWsize + embSsize + n_feat
      lstm_out_size = hyperparams['lstm_out_size']
      num_layers = hyperparams.get('lstm_num_layers', 1)
      self.lstm = nn.LSTM(lstm_in_size, lstm_out_size, num_layers=num_layers,bidirectional=True, batch_first=True)

      linear_out_size = hyperparams['linear_out_size']
      self.linear = nn.Linear(2*lstm_out_size, linear_out_size)
      self.out = nn.Linear(linear_out_size, n_labels)

   def forward(self, lw, w, s, f):
      x = self.embLW(lw)
      y = self.embW(w)
      z = self.embS(s)
      x = self.dropLW(x)
      y = self.dropW(y)
      z = self.dropS(z)

      x = torch.cat((x, y, z, f), dim=2)
      x = self.lstm(x)[0]

      activation = self.activation.lower()
      if activation == 'relu':
         x = func.relu(x)
      elif activation == 'tanh':
         x = func.tanh(x)
      elif activation == 'sigmoid':
         x = func.sigmoid(x)

      x = self.linear(x)
      x = self.out(x)
      return x
   


