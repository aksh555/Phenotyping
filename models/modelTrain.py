import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
import tensorflow as tf
import numpy as np
from dataproc import extract_wvs
from constants import *

from math import floor
import random
import sys
import time
from transformers import BertTokenizer, BertForSequenceClassification

class BaseModel(nn.Module):

    def __init__(self, numClasses, embed_file,  dropout=0.2, embed_size=50, vocab_size=500):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.numClasses=numClasses
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        #make embedding layer
        if embed_file:
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            print(vocab_size,embed_size)
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            nn.init.xavier_uniform_(self.embed.weight)
        self.embed_size=self.embed.weight.size()[1]

class BERTLATA(BaseModel):
    def __init__(self,numClasses, embed_file,num_filter_maps=256,embed_size=50, dropout=0.2,vocab_size=500):
        super(BERTLATA, self).__init__( numClasses=numClasses, embed_file=embed_file,dropout=dropout, embed_size=embed_size,vocab_size=vocab_size)
        numberofLayers=4
        self.config = BertConfig(vocab_size=self.embed.weight.size()[0], 
        hidden_size=self.embed.weight.size()[1],
        max_position_embeddings=MAX_LENGTH,
        num_hidden_layers=numberofLayers,
        num_attention_heads=numberofLayers,
        intermediate_size=num_filter_maps,
        output_hidden_states=True)
        self.encoder = BertModel(config=self.config)
        self.U = nn.ModuleList([nn.Linear(self.embed.weight.size()[1], self.numClasses) for i in range(numberofLayers+2)])
        self.final=nn.Linear(self.embed.weight.size()[1]*(numberofLayers+2),self.numClasses)

    def forward(self, x,labels=None):
        outputs = self.encoder(input_ids=x)
        attentionOutput=[]
        if(len(outputs)==3):
                attentionOutput=outputs[2]
        elif(len(outputs)==2 ):
                attentionOutput=outputs[1]
        m=[]
        Attentions=[]
        for i, output in enumerate(attentionOutput):
            alpha = F.softmax(self.U[i].weight.matmul(output.transpose(1,2)), dim=2)
            Attentions.append(alpha)
            m1 = alpha.matmul(output)
            m1 = self.dropout(m1)
            m.append(m1)
        alpha = F.softmax(self.U[-1].weight.matmul(outputs[0].transpose(1,2)), dim=2)
        Attentions.append(alpha)
        m1 = alpha.matmul(outputs[0])
        m1 = self.dropout(m1)
        m.append(m1)
        m=torch.cat(m,-1)
        y=self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        yhat = y
        return yhat, Attentions

class ENCAML(BaseModel):
    def __init__(self,numClasses, embed_file,num_filter_maps=256,embed_size=50, dropout=0.2,vocab_size=500):
        super(ENCAML, self).__init__( numClasses=numClasses, embed_file=embed_file,dropout=dropout, embed_size=embed_size,vocab_size=vocab_size)
        kernel_size=[3,5,7,9]
        self.conv=nn.ModuleList([nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=i, padding=int(floor(i/2))) for i in kernel_size])
        self.U = nn.ModuleList([nn.Linear(num_filter_maps, self.numClasses) for i in range(len(kernel_size))])
        self.final=nn.Linear(num_filter_maps*len(kernel_size),self.numClasses)
        
    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        m=[]
        Attentions=[]
        for i, conv in enumerate(self.conv):
            x1 = torch.tanh(conv(x).transpose(1,2))
            alpha = F.softmax(self.U[i].weight.matmul(x1.transpose(1,2)), dim=2)
            Attentions.append(alpha)
            m1 = alpha.matmul(x1)
            m1 = self.dropout(m1)
            m.append(m1)
        m=torch.cat(m,-1)
        y=self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        yhat = y
        return yhat, Attentions

        
#Create a vocabulary file
#criterion=nn.MultiLabelSoftMarginLoss()
#ind2w = {i+2:w for i,w in enumerate(sorted(vocab))}
#ind2w[0]=PAD_TOKEN
#ind2w[1]=UNK_TOKEN
#w2ind = {w:i for i,w in ind2w.items()}
#model=BERTLATA(numClasses,None,64,128,0.2,vocabsize)
#generate batchwise data and target( multi label), in the loop of epochs train model
#outputs, attentions = model(data)
#loss = criterion(outputs, target)
