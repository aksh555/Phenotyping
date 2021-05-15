import sys
sys.path.append('../')
sys.path.append('./')
from collections import defaultdict
import csv
import math
import numpy as np
import sys
from constants import *
import models
import torch
import torch
import torch.nn as nn
from datagen import *
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#__________________________________________________________

import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import os 
import numpy as np
import operator
import random
import sys
import time
from constants import *
import evaluation
import persistence
from pytorch_model_summary import summary

#****************************************************************

vocab = set()
vocab_file="%s/vocab.csv" % DATA_DIR
codes_file="%s/TOP_2194_CODES.csv" % DATA_DIR
embed_size=128
filter_maps=64
drop_out=0.3
#if GPU is there set it to True
#gpu=True
gpu=False
with open(vocab_file, 'r') as vocabfile:
    for i,line in enumerate(vocabfile):
        line = line.rstrip()
        if line != '':
            vocab.add(line.strip())
    ind2w = {i+2:w for i,w in enumerate(sorted(vocab))}
    ind2w[0]=PAD_TOKEN
    ind2w[1]=UNK_TOKEN
    w2ind = {w:i for i,w in ind2w.items()}

# As the labels were in text format below code is used to map codes to numbers
# this could be skipped, mainly include length of labels in the dictionary
# update data generation in case yo  directly have on hot representaiton
#if its not multi-label you can directly give label and update the to cross entropy loss than
#multi label loss. and use softmax in test method instead of signmoid and argmax instead of round
codes = set([])
with open(codes_file, 'r') as labelfile:
    lr = csv.reader(labelfile)
    for i,row in enumerate(lr):
        codes.add(row[0])
    ind2c = {i:c for i,c in enumerate(sorted(codes))}
    c2ind = {c:i for i,c in ind2c.items()}
dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind}

#model=models.BERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.AlBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.DebertaBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.DebertaBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.DistilBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.XLNetBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
model=models.RoBERTaBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.MobileBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.MobileBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))
#model=models.AlBERT(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))

#model=models.enCAML(len(ind2c),None, filter_maps,embed_size,drop_out,len(ind2w))

data = Variable(torch.LongTensor(np.array([[0]])))
print(summary(model, data, show_input=True,show_hierarchical=True))
if(gpu):
    model.cuda()

#this includes the file name of train file. due to large size i used test only for training for 
#Replace this with training file name 
data_path="%s/trainD_Database.csv" % DATA_DIR
#maximum epoch , anyways the best model uptil 10 patitientce is saved, so it may not go till max epoch
dicts['n_epochs']=100
dicts['batch_size']=32
dicts['data_path']=data_path
dicts['criterion']='prec_at_8'
dicts['patience']=10

dicts['params']={}
dicts['gpu']=False
#GPU is available set this to True
#dicts['gpu']=True
dicts['optimizer']=optim.Adam(model.parameters(), weight_decay=0, lr=1e-3)

#if only used for testing then the full path name could be provided to this key
#dicts['test_model']='./saved_models/RoBERTaBERT_Feb_08/model_best_prec_at_8.pth'
if dicts['test_model']:
    sd = torch.load(dicts['test_model'])
    model.load_state_dict(sd)
    if gpu:
        model.cuda()

#****************************************************************

def train_epochs( model,dicts):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    
    n_epochs=dicts['n_epochs']
    test_only=dicts['test_model'] != ''
    gpu=dicts['gpu']
    criterion=dicts['criterion']
    patience=dicts['patience']
    
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(n_epochs):
        #only test on train/test set on very last epoch
        if epoch == 0 and not test_only:
            model_dir = os.path.join(MODEL_DIR, '_'.join([model.__class__.__name__, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.mkdir(model_dir)
        elif dicts['test_model']:
            model_dir = os.path.dirname(os.path.abspath(dicts['test_model']))
        
        metrics_all = one_epoch(model, epoch, dicts, model_dir, test_only)
        
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        if test_only:
            break

        #save metrics, model, params
        persistence.save_everything(gpu, metrics_hist_all, model, model_dir, criterion)

        if criterion in metrics_hist.keys():
            if early_stop(metrics_hist, criterion, patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (criterion, patience))
                test_only = True
                dicts['test_model'] = '%s/model_best_%s.pth' % (model_dir, criterion)
                if dicts['test_model']:
                    sd = torch.load(dicts['test_model'])
                    model.load_state_dict(sd)
                if gpu:
                    model.cuda()

    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev': 
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False
        
def one_epoch(model, epoch, dicts, model_dir, testing):
    """
        Wrapper to do a training epoch and test on dev
    """
    n_epochs=dicts['n_epochs']
    print("testing",testing)
    if not testing:
        losses = train(model,  epoch, dicts )
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
    else:
        loss = np.nan
        if gpu:
            model.cuda()

    fold = 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True

    #test on dev
    metrics = test(model, epoch,  fold, dicts, model_dir, testing)
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(model, epoch, "test", dicts, model_dir, True)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    return metrics_all


def train(model, epoch, dicts):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    print("EPOCH %d" % epoch)
    num_labels = len(dicts['ind2c'])
    
    criterion=nn.MultiLabelSoftMarginLoss()
    criterionCross=nn.CrossEntropyLoss()

    losses = []
    #how often to print some info to stdout
    print_every = 1000

    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    data_path=dicts['data_path']
    batch_size=dicts['batch_size']
    optimizer=dicts['optimizer']

    model.train()
    gen = data_generator(data_path, dicts, batch_size, num_labels)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, _ = tup
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        if gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()

        outputs, attentions = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

        if batch_idx % print_every == 0:
            #print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-10:])))
    return losses

def test(model,epoch, fold,  dicts, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """
    #use here train Ensure that the train file and test file differ by only fold ex train_database, dev_database, test_database
    #so  that replace gives the testing file name
    #filename = data_path.replace('train', fold)
    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])

    #if the valus are categorical then use cross entropy loss instead of multu label loss
    criterion=nn.MultiLabelSoftMarginLoss()

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    model.eval()
    with torch.no_grad():
        gen = data_generator(filename, dicts, 1, num_labels)
        for batch_idx, tup in tqdm(enumerate(gen)):
            data, target, hadm_ids= tup
            data, target= Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
            if gpu:
                data = data.cuda()
                target = target.cuda()
            model.zero_grad()
            output, attns = model(data)
            output = torch.sigmoid(output)
            loss = criterion(output, target)
            output = output.data.cpu().numpy()
            yhat_raw.append(output)
            #use argmax instead of round if catetgoritcal 
            output=np.round(output)
            losses.append(loss.data.item())
            target_data = target.data.cpu().numpy()
            #save predictions, target, hadm ids
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)
            hids.extend(hadm_ids)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    if(not testing):
        preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 10 else [5,8,10,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    return metrics

epochs_trained = train_epochs( model, dicts)





