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
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from focal_loss.focal_loss import FocalLoss
import gensim
from gensim.models import Word2Vec


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
    def __init__(self, numClasses, embed_file=None, embed_size=128, dropout=0.2, vocab_size=500):
        super(BERT, self).__init__(numClasses=numClasses, embed_file=embed_file,
                                   dropout=dropout, embed_size=embed_size, vocab_size=vocab_size)
        numberofLayers = 4
        self.config = BertConfig(vocab_size=self.embed.weight.size()[0],
                                 hidden_size=self.embed.weight.size()[1],
                                 max_position_embeddings=MAX_LEN,
                                 num_hidden_layers=numberofLayers,
                                 num_attention_heads=numberofLayers,
                                 intermediate_size=3072,
                                 output_hidden_states=True)

        # self.embed_size = embed_size
        # self.numClasses = numClasses
        self.encoder = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", config=self.config)
        # n c l
        self.conv = nn.Conv1d(768,768,kernel_size=3, padding=1)
        self.final = nn.Linear(embed_size, numClasses)
        self.mheadattn = nn.MultiheadAttention(self.embed_size, 4)
        self.dropout = nn.Dropout(dropout)
        self.m = nn.ReLU()

    def forward(self, notes, attention_masks, labels=None):
        fin_outputs, feats = [], []
        assert len(notes)>0
        for x,attention_mask in zip(notes,attention_masks):
            x = x.cuda()
            # print(x.shape)
            attention_mask = attention_mask.cuda()
            outputs = self.encoder(input_ids=x, attention_mask=attention_mask)[0]
            
            fin_outputs.append(outputs.squeeze(0))
            
            # attention_mask = ~attention_mask.bool()
            #print(attention_mask)
            # outputs = outputs.permute(1,0,2)
            # # print('enc', outputs.shape)
            # attn_output, attn_output_weights = self.mheadattn(outputs, outputs, outputs, key_padding_mask=attention_mask)
            # attn_output = attn_output.permute(1,0,2)
            # m = torch.mean(attn_output, dim=1).cuda()
            # m = m.squeeze(0)
            # attn_outputs.append(attn_output.squeeze(0))
            # feats.append(m)
        
        fin_output = torch.stack(fin_outputs)
        outputs = torch.mean(fin_output, dim=0)
        outputs = outputs.unsqueeze(0)
        outputs = outputs.permute(1,0,2)
        # print('out', outputs.shape)
        attn_output, attn_output_weights = self.mheadattn(outputs, outputs, outputs)
        attn_output = attn_output.permute(1,0,2)
        m = torch.mean(attn_output, dim=1).cuda()
        m = m.squeeze(0)
        
        # fin_attn_output = torch.stack(attn_outputs)
        # # print('fin_attn', fin_attn_output.shape)
        # fin_attn_output = torch.mean(fin_attn_output, dim=0)
        # fin_feats = torch.stack(feats)
        # fin_feats = torch.mean(fin_feats, dim=0)
        # # print('fin_attn2', fin_attn_output.shape)
        # fin_attn_output = fin_attn_output.unsqueeze(0)
        fin_attn_output = attn_output.permute(0,2,1)
        fin_attn_output = self.conv(fin_attn_output)
        fin_attn_output = fin_attn_output.permute(0,2,1)
        fin = self.final(fin_attn_output)
        fin = torch.mean(fin, dim=1)
        
        return fin, m

BATCH_SIZE = 1
MAX_LEN = 512
numClasses = 10
EPOCHS = 10
CHUNK_SIZE  = 510

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

        input_ids, attention_masks = [], []
        note_words = review.split()
        
        k = len(note_words)//CHUNK_SIZE

        if k == 0:
            encoding = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=len(note_words),
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
                padding=True
            )
            input_ids.append(encoding['input_ids'].flatten())
            attention_masks.append(encoding['attention_mask'].flatten())
        else:
            count = 0
            for i in range(CHUNK_SIZE,len(note_words),CHUNK_SIZE):
                if count == k:
                    break
                chunk = " ".join(note_words[i-CHUNK_SIZE:i])
                encoding = self.tokenizer.encode_plus(
                    chunk,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True,
                    padding=True
                )
                input_ids.append(encoding['input_ids'].flatten())
                attention_masks.append(encoding['attention_mask'].flatten())
                count+=1
        
        return {
            'notes_text': review,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'targets': torch.tensor(target, dtype=torch.float)
        }


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model.cuda()
    model = model.train()
    losses = []
    tr_loss = 0.

    for d in data_loader:
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
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
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
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
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"].cuda()

            outputs, _ = model(input_ids,attention_mask)
            # print(outputs.shape)
            preds = F.sigmoid(outputs) > 0.5
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
        pred_bools = [pl > 0.50 for pl in pred_labels]
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
    args = parser.parse_args()

    data_dir = "./mimic-iii"
    files_dir = "./dump_files"
    model_dir = "./models"
    out_dir = "./out"

    df = pd.read_csv(os.path.join(files_dir, 'pheno_disch-ref.csv'))
    tz = BertTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", padding=True, truncation=True)
    model = BERT(numClasses, embed_size=768, vocab_size=tz.vocab_size)
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
        num_training_steps=total_steps
    )

    l = df.sum(axis=0).tolist()[2:]
    wts = [1-(i/df.shape[0]) for i in l]
    #loss_fn = nn.MultiLabelSoftMarginLoss(weight=torch.FloatTensor(wts).cuda())
    loss_fn = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(wts).cuda())
    #criterion = FocalLoss(alpha=10, gamma=5)
    history = defaultdict(list)
    best_loss = float('inf')

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
                    model_dir, 'clinical-bert-chunks-conv-disch-ref.bin'))
                best_loss = val_loss

    if args.do_eval:
        model.load_state_dict(torch.load(
            os.path.join(model_dir, 'clinical-bert-chunks-conv-disch-ref.bin')))
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
            out_dir, 'CR_clinical-bert-chunks-conv-disch-ref.txt'), 'wb'))  # save report
        print(clf_report)
        with open(os.path.join(out_dir, 'res-clinical-bert-chunks-conv-disch-ref.txt'), 'w') as f:
            f.write(outs)
        print('jaccard micro',jaccard_score(y_test, y_pred, average='micro'))
        print('jaccard macro', jaccard_score(y_test, y_pred, average='macro'))
        print('ham',hamming_loss(y_test, y_pred))
        print('auc',roc_auc_score(y_test, y_pred, average='micro'))


if __name__ == '__main__':
    main()