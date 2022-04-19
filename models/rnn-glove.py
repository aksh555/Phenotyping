import torch   
import torchtext
#handling text data
from torchtext.legacy import data
import spacy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
import numpy as np
from math import floor
import random
import sys
import pickle
import time
import os
from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import jaccard_score, roc_auc_score, confusion_matrix, hamming_loss, multilabel_confusion_matrix, f1_score, accuracy_score, classification_report
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from transformers import AutoTokenizer, AutoModel
from torch.optim import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from focal_loss.focal_loss import FocalLoss
import gensim
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
            
        #Constructor
        super().__init__()          
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                        hidden_dim, 
                        num_layers=n_layers, 
                        bidirectional=bidirectional, 
                        dropout=dropout,
                        batch_first=True)
        
        self.fc_dim = 128
        self.embed_size = 300
        #dense layer
        self.fc = nn.Linear(self.fc_dim * 2, 10)
        self.fc1 = nn.Linear(self.embed_size, self.fc_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, self.fc_dim)
        self.conv = nn.Conv1d(self.embed_size,self.embed_size,kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.mp = nn.MaxPool1d(1, padding=0)
        #activation function
        self.act = nn.Sigmoid()
        self.mheadattn = nn.MultiheadAttention(self.embed_size, 4)
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text).cuda()
        
        #embedded = [batch size, sent_len, emb dim]
        embedded = embedded.permute(1,0,2)
        embedded, attn_output_weights = self.mheadattn(embedded, embedded, embedded)
        embedded = embedded.permute(1,0,2)
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True).cuda()
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).cuda()
                
        #hidden = [batch size, hid dim * num directions]
        dense_op = self.fc2(hidden)

        embedded = embedded.permute(0,2,1)
        conv_op = self.conv(embedded)
        conv_op = self.mp(conv_op)
        conv_op = conv_op.permute(0,2,1)

        fin1 = self.fc1(conv_op)
        fin1 = torch.mean(fin1, dim=1)
        fin1 = self.act(fin1)

        #Final activation function
        outputs = self.fc(torch.cat((dense_op, fin1), dim=1))
        # outputs = self.act(outputs)
        
        return outputs


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples, labels):
    model.cuda()
    model = model.train()
    losses = []
    tr_loss = 0.

    for d in data_loader:
        text, text_lengths = d.TEXT  
        
        #convert to 1D tensor
        predictions = model(text, text_lengths)
        y = torch.cat([ getattr(d, feat).unsqueeze(1) for feat in labels ], dim=1).float()
        # print(y.size())
        #compute the loss
        
        loss = loss_fn(predictions, y)

        loss.backward()
        tr_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return tr_loss / n_examples


def eval_model(model, data_loader, loss_fn, n_examples, labels):
    model.cuda()
    model = model.eval()
    tr_loss = 0

    with torch.no_grad():
        for d in data_loader:
            text, text_lengths = d.TEXT   
        
            #convert to 1D tensor
            predictions = model(text, text_lengths)
            y = torch.cat([ getattr(d, feat).unsqueeze(1) for feat in labels ], dim=1).float()
            #compute the loss
            loss = loss_fn(predictions, y)
            tr_loss += loss.item()

    return tr_loss / n_examples


def get_predictions(model, data_loader, labels):
    model.cuda()
    model = model.eval()

    predictions = []
    real_values = []
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for d in data_loader:
            text, text_lengths = d.TEXT   
        
            outputs = model(text, text_lengths)
            
        
            # print(outputs.shape)
            preds = F.sigmoid(outputs) > 0.5
            pred_label = torch.sigmoid(outputs)
            targets = torch.cat([ getattr(d, feat).unsqueeze(1) for feat in labels ], dim=1).float()
            pred_label = pred_label.cpu().numpy()
            targets = targets.cpu().numpy()
            true_labels.append(targets)
            pred_labels.append(pred_label)

            # Converting flattened binary values to boolean values

            preds = preds.long()
            preds = preds.data.cpu().numpy()
            # print(preds.shape)
            predictions.extend(preds)
            real_values.extend(targets)
        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]
        true_bools = [tl == 1 for tl in true_labels]
        # boolean output after thresholding
        pred_bools = [pl > 0.50 for pl in pred_labels]
        #print(len(pred_bools), len(true_bools))

    #predictions = torch.stack(predictions).cpu()
    #real_values = torch.stack(real_values).cpu()
    return predictions, real_values, pred_bools, true_bools

class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            
            if self.y_vars is not None: # we will concatenate y into a single tensor
                y = torch.cat([ getattr(batch, feat).unsqueeze(1) for feat in self.y_vars ], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)
    
    def __len__(self):
        return len(self.dl)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--model", default="clinical")
    args = parser.parse_args()

    TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float,batch_first=True)


    data_dir = "./mimic-iii"
    files_dir = "./dump_files"
    model_dir = "./models"
    out_dir = "./out"

    df = pd.read_csv(os.path.join(files_dir, 'pheno_disch-ref.csv'))

    fields = [(None, None), ('TEXT', TEXT)]
    for l in df.columns[2:]: 
        fields.append((l, LABEL))

    training_data = data.TabularDataset(path = './dump_files/pheno_disch-ref.csv',format = 'csv',fields = fields,skip_header = True)

    #initialize glove embeddings
    TEXT.build_vocab(training_data,min_freq=3,vectors = "glove.840B.300d")  
    LABEL.build_vocab(training_data)

    #No. of unique tokens in text
    print("Size of TEXT vocabulary:",len(TEXT.vocab))

    #No. of unique tokens in label
    print("Size of LABEL vocabulary:",len(LABEL.vocab))
    
    
    size_of_vocab = len(TEXT.vocab)
    embedding_dim = 300
    num_hidden_nodes = 256
    num_output_nodes = 1
    num_layers = 2
    bidirection = True
    dropout = 0.1
    BATCH_SIZE = 1
    RANDOM_SEED = 42
    EPOCHS = 50

    train_data, test_data = training_data.split(split_ratio=0.1, random_state = random.seed(RANDOM_SEED))
    valid_data, test_data = test_data.split(split_ratio=0.5, random_state = random.seed(RANDOM_SEED))

    #Load an iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.TEXT),
        sort_within_batch=True,
        device = 'cuda')
    
    txt_col = 'TEXT'
    # train_iterator = BatchWrapper(train_iterator, txt_col, df.columns[2:])
    # valid_iterator = BatchWrapper(valid_iterator, txt_col, df.columns[2:])
    # test_iterator = BatchWrapper(test_iterator, txt_col, None)

    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                    bidirectional = True, dropout = dropout)

    #Initialize the pretrained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    # RANDOM_SEED = 42

    df_train, df_test = train_test_split(
        df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(
        df_test, test_size=0.5, random_state=RANDOM_SEED)

    # train_data_loader = create_data_loader(df_train, tz, MAX_LEN, BATCH_SIZE)
    # val_data_loader = create_data_loader(df_val, tz, MAX_LEN, BATCH_SIZE)
    # test_data_loader = create_data_loader(df_test, tz, MAX_LEN, BATCH_SIZE)
    test_label_cols = df_test.columns[2:]

    # optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    optimizer = Adam(model.parameters(), lr=5e-5)
    total_steps = len(train_iterator) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=total_steps,
        
    )

    l = df.sum(axis=0).tolist()[2:]
    wts = [1-(i/df.shape[0]) for i in l]
    # loss_fn = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor(wts).cuda())
    loss_fn = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(wts).cuda())
    #criterion = FocalLoss(alpha=10, gamma=5)
    history = defaultdict(list)
    best_loss = float('inf')
    stop_it = 0
    if args.do_train:
        for epoch in range(EPOCHS):

            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)

            train_loss = train_epoch(
                model,
                train_iterator,
                loss_fn,
                optimizer,
                scheduler,
                len(df_train),
                df.columns[2:]
            )

            print(f'Train loss {train_loss}')

            val_loss = eval_model(
                model,
                valid_iterator,
                loss_fn,
                len(df_val),
                df.columns[2:]
            )

            print(f'Val loss {val_loss}')
            print()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(
                    model_dir, f'exp-conv-disch-ref.bin'))
                best_loss = val_loss
                stop_it = 0
            else:
                stop_it += 1
                if stop_it == 3:
                    print(f'Early Stop: No improvement in validation loss')
                    break
        print('Training done!')

    if args.do_eval:
        model.load_state_dict(torch.load(
            os.path.join(model_dir, f'exp-conv-disch-ref.bin')))
        y_pred, y_test, pred_bools, true_bools = get_predictions(model, test_iterator, df.columns[2:])
        # print(y_pred)
        # print(y_test)

        # get features
        # get_feats(model, train_data_loader, val_data_loader, test_data_loader)

        outs = ""
        outs += f"jaccard_micro: {jaccard_score(y_test, y_pred, average='micro')}\n"
        outs += f"jaccard_macro: {jaccard_score(y_test, y_pred, average='macro')}\n"
        outs += f"hammingloss: {hamming_loss(y_test, y_pred)}\n"
        outs += f"F1: {f1_score(true_bools, pred_bools,average='micro')}\n"
        outs += f"Acc: {accuracy_score(true_bools, pred_bools)}\n"
        outs += f"auc: {roc_auc_score(y_test, y_pred, average='micro')}\n"
        # Print and save classification report
        print('F1 score: ', f1_score(
            true_bools, pred_bools, average='micro'))
        print('Flat Accuracy: ', accuracy_score(
            true_bools, pred_bools), '\n')
        clf_report = classification_report(
            true_bools, pred_bools, target_names=test_label_cols)
        pickle.dump(clf_report, open(os.path.join(
            out_dir, 'CR_clinical-bert-conv-disch-ref.txt'), 'wb'))  # save report
        print(clf_report)
        with open(os.path.join(out_dir, 'RES-clinical-bert-conv-disch-ref.txt'), 'w') as f:
            f.write(outs)
        print('jaccard micro',jaccard_score(y_test, y_pred, average='micro'))
        print('jaccard macro', jaccard_score(y_test, y_pred, average='macro'))
        print('ham',hamming_loss(y_test, y_pred))
        print('auc',roc_auc_score(y_test, y_pred, average='micro'))


if __name__ == '__main__':
    main()