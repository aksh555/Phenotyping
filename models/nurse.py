import torch
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

class BaseModel(nn.Module):
    def __init__(self, numClasses, embed_file=None,  dropout=0.2, embed_size=50, vocab_size=500):
        super(BaseModel, self).__init__()
        # torch.manual_seed(1337)
        self.numClasses = numClasses
        self.embed_size = embed_size
        self.dropout = nn.Dropout(p=dropout)
        # make embedding layer

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight)
        # self.embed_size = self.embed.weight.size()[1]


class BERT(BaseModel):
    def __init__(self, args, numClasses, embed_file=None, num_filter_maps=256, embed_size=128, dropout=0.2, vocab_size=500):
        super(BERT, self).__init__(numClasses=numClasses, embed_file=embed_file,dropout=dropout, embed_size=embed_size, vocab_size=vocab_size)
        numberofLayers = 4
        self.config = BertConfig(vocab_size=self.embed.weight.size()[0],
                                 hidden_size=self.embed.weight.size()[1],
                                 max_position_embeddings=MAX_LEN,
                                 num_hidden_layers=numberofLayers,
                                 num_attention_heads=numberofLayers,
                                 intermediate_size=num_filter_maps,
                                 output_hidden_states=True)

        self.encoder = BertModel.from_pretrained(args.model, config=self.config)
        self.conv1 = nn.Conv1d(768,768,kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768,768,kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(768,768,kernel_size=3, padding=1)

        self.rnn_hidden_size = 128
        kernel_size=[1]
        self.convlist=nn.ModuleList([nn.Conv1d(self.embed_size, self.embed_size, kernel_size=i, padding=int(floor(i/2))) for i in kernel_size])
        self.bigru = nn.GRU(input_size=self.embed_size,
                          hidden_size=self.rnn_hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.final = nn.Linear(self.embed_size, self.numClasses)
        # self.final = nn.Linear(self.numClasses*self.embed_size, self.numClasses)
        # self.final = nn.Linear(2*self.rnn_hidden_size, self.numClasses)
        self.mheadattn = nn.MultiheadAttention(self.embed_size, 4)
        self.act = nn.ReLU()
        self.mp = nn.MaxPool1d(1, padding=0)
        self.avgpool = nn.AvgPool1d(1, padding=0)
        
    def forward(self, x, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.encoder(input_ids=x, attention_mask=attention_mask)[0]
            # outputs = torch.mean(torch.stack(outputs.hidden_states), dim=0)
            # print('attn mask', attention_mask.shape)
            attention_mask = ~attention_mask.bool()
            #print(attention_mask)
            outputs1 = outputs.permute(1,0,2)
        # print('enc', outputs.shape)
        
        attn_output, attn_output_weights = self.mheadattn(outputs1, outputs1, outputs1, key_padding_mask=attention_mask)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.permute(1,0,2)
        attn_output = torch.add(attn_output,outputs)
        # skip_initial = outputs

        # m = torch.mean(attn_output, dim=1).cuda()
        # m = m.squeeze(0)
        #print('attn op', attn_output.shape)
        
        # attn_output = attn_output.permute(0,2,1)
        # concat_x = []
        # for conv in self.convlist:
        #     x = conv(attn_output)
        #     x = x.permute(0,2,1)
        #     concat_x.append(x)
        # concat_x = torch.cat(concat_x,dim=2)
        # # print('con_x', concat_x.shape)

        # rnn_inputs = attn_output.cuda()
        # outputs, last_hidden = self.bigru(rnn_inputs)
        # print('lh', outputs.shape)
        # last_hidden = self._bridge_bidirectional_hidden(last_hidden).cuda()
        # print('lh', last_hidden.shape)

        # SKIP
        # print(attn_output.shape)
        attn_output = attn_output.permute(0,2,1)
        # skip0 = attn_output
        attn_output = self.conv1(attn_output)
        # attn_output = torch.add(skip0,attn_output)
        
        # attn_output = self.mp(attn_output)
        # attn_output = attn_output.permute(0,2,1)
        # attn_output = self.dropout(attn_output)

        # skip2 = attn_output
        # attn_output = self.conv2(attn_output)
        # attn_output = torch.add(skip2,attn_output)
        # attn_output = self.mp(attn_output)

        # skip3 = attn_output
        # attn_output = self.conv3(attn_output)
        # attn_output = torch.add(skip3,attn_output)
        # attn_output = self.mp(attn_output)

        # attn_output = torch.add(attn_output,skip_initial.permute(0,2,1))
        # attn_output = self.conv(attn_output)
        
        attn_output = self.avgpool(attn_output)
        # # attn_output = self.dropout(attn_output)

        # attn_output = self.conv3(attn_output)
        
        # attn_output = self.avgpool(attn_output)
        # attn_output = torch.mean(attn_output, dim=2)
        attn_output = attn_output.permute(1,0,2)
        # print(attn_output.shape)
        # # print(outputs.shape)
        
        fin = self.final(outputs)
        # # fin = self.final(last_hidden)
        fin = torch.mean(fin, dim=1)
        # #print('m', m.shape)
        # return attn_output, None
        return fin, None
    
    @staticmethod
    def _bridge_bidirectional_hidden(hidden):
        """
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)

BATCH_SIZE = 1
MAX_LEN = 512
numClasses = 10
EPOCHS = 100


class NotesDataset(Dataset):

    def __init__(self, notes, targets, tokenizer, max_len):
        self.notes = notes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, item):
        review = str(self.notes[item])
        target = self.targets[item]
        
        #print(target)
        if review == None:
            print('err')
            sys.exit()

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            
        )

        return {
            'notes_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model.cuda()
    model = model.train()
    losses = []
    tr_loss = 0.

    for d in data_loader:
        input_ids = d["input_ids"].cuda()
        attention_mask = d["attention_mask"].cuda()
        targets = d["targets"].cuda()

#         print('input', input_ids.shape)
#         print(input_ids)
#         print("target", targets.shape)
#         print(targets)
        outputs, _ = model(input_ids,attention_mask)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        loss.backward()
        tr_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return tr_loss / n_examples


def eval_model(model, data_loader, loss_fn, n_examples):
    model.cuda()
    model = model.eval()
    tr_loss = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = d["targets"].cuda()

            outputs, _ = model(input_ids,attention_mask)

            loss = loss_fn(outputs, targets)
            tr_loss += loss.item()

    return tr_loss / n_examples


def get_predictions(model, data_loader):
    model.cuda()
    model = model.eval()

    predictions = []
    real_values = []
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for d in data_loader:
            texts = d["notes_text"]
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = d["targets"].cuda()

            outputs, _ = model(input_ids,attention_mask)
            # print(outputs.shape)
            preds = F.sigmoid(outputs) >= 0.4
            pred_label = torch.sigmoid(outputs)
            targets = targets.long()
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
        pred_bools = [pl >= 0.40 for pl in pred_labels]
        #print(len(pred_bools), len(true_bools))

    #predictions = torch.stack(predictions).cpu()
    #real_values = torch.stack(real_values).cpu()
    return predictions, real_values, pred_bools, true_bools


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NotesDataset(
        notes=df.TEXT.to_numpy(),
        targets=df.iloc[:, 2:].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True
    )


def get_feats(model, train_data_loader, valid_data_loader, test_data_loader):
    model = model.eval()
    train_feats = []
    valid_feats = []
    test_feats = []
    
    with torch.no_grad():
        for d in train_data_loader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = np.array(d["targets"])[0]
            
            _, feats_train = model(input_ids,attention_mask)
            feats_train = feats_train.data.cpu().numpy()
            feats_train = np.concatenate([feats_train, targets], axis=0)
            train_feats.append(feats_train)
            
        for d in valid_data_loader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = np.array(d["targets"])[0]

            _, feats_valid = model(input_ids,attention_mask)
            feats_valid = feats_valid.data.cpu().numpy()
            feats_valid = np.concatenate([feats_valid, targets], axis=0)
            valid_feats.append(feats_valid)
            
        for d in test_data_loader:
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            targets = np.array(d["targets"])[0]

            _, feats_test = model(input_ids,attention_mask)
            feats_test = feats_test.data.cpu().numpy()
            feats_test = np.concatenate([feats_test, targets], axis=0)
            test_feats.append(feats_test)
            
    train_feats = np.array(train_feats)
    valid_feats = np.array(valid_feats)
    test_feats = np.array(test_feats)
    df_tr = pd.DataFrame(train_feats)
    df_val = pd.DataFrame(valid_feats)
    df_test = pd.DataFrame(test_feats)
    df_tr.to_csv("./dump_files/train_pheno_disch_feats.csv")
    df_val.to_csv("./dump_files/valid_pheno_disch_feats.csv")
    df_test.to_csv("./dump_files/test_pheno_disch_feats.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--model", default="clinical")
    args = parser.parse_args()

    model_map = {
        "clinical": "emilyalsentzer/Bio_ClinicalBERT",
        "biomed": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "blue" : "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        "biobert": "dmis-lab/biobert-base-cased-v1.1"
    }
    args.model_name = args.model
    args.model = model_map[args.model]

    data_dir = "./mimic-iii"
    files_dir = "./dump_files"
    model_dir = "./models"
    out_dir = "./out"

    df = pd.read_csv(os.path.join(files_dir, 'pheno_nurse-ref.csv'))
    tz = BertTokenizer.from_pretrained(
        args.model, padding=True, truncation=True)
    model = BERT(args,numClasses, embed_size=768, vocab_size=tz.vocab_size, num_filter_maps=3072)
    # for p in model.parameters():
    #     if p.requires_grad:
    #         print(p.name)

    RANDOM_SEED = 42

    df_train, df_test = train_test_split(
        df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(
        df_test, test_size=0.5, random_state=RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tz, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tz, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tz, MAX_LEN, BATCH_SIZE)
    test_label_cols = df_test.columns[2:]

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=total_steps,
        
    )

    l = df.sum(axis=0).tolist()[2:]
    print(l)
    wts = [1-(i/df.shape[0]) for i in l]
    #loss_fn = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor(wts).cuda())
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
                train_data_loader,
                loss_fn,
                optimizer,
                scheduler,
                len(df_train)
            )

            print(f'Train loss {train_loss}')

            val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn,
                len(df_val)
            )

            print(f'Val loss {val_loss}')
            print()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(
                    model_dir, f'nurse-single-note-with-annotations.bin'))
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
            os.path.join(model_dir, f'nurse-single-note-with-annotations.bin')))
        y_pred, y_test, pred_bools, true_bools = get_predictions(
            model, test_data_loader)
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
            out_dir, 'nurse-single-note-with-annotations.txt'), 'wb'))  # save report
        print(clf_report)
        with open(os.path.join(out_dir, 'nurse-single-note-with-annotations.txt'), 'w') as f:
            f.write(outs)
        print('jaccard micro',jaccard_score(y_test, y_pred, average='micro'))
        print('jaccard macro', jaccard_score(y_test, y_pred, average='macro'))
        print('ham',hamming_loss(y_test, y_pred))
        print('auc',roc_auc_score(y_test, y_pred, average='micro'))


if __name__ == '__main__':
    main()