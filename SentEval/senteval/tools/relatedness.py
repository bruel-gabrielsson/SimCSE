# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Semantic Relatedness (supervised) with Pytorch
"""
from __future__ import absolute_import, division, unicode_literals

import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim

from scipy.stats import pearsonr, spearmanr

import logging

from tqdm import tqdm



class RelatednessPytorch(object):
    # Can be used for SICK-Relatedness, and STS14
    def __init__(self, train, valid, test, devscores, config):
        # fix seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        torch.cuda.manual_seed(config['seed'])

        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        self.inputdim = train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.l2reg = 0.
        self.batch_size = 64
        self.maxepoch = 1000
        self.early_stop = True

        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
            nn.Softmax(dim=-1),
        )
        self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2reg)

    def prepare_data(self, trainX, trainy, devX, devy, testX, testy):
        # Transform probs to log-probs for KL-divergence
        trainX = torch.from_numpy(trainX).float().cuda()
        trainy = torch.from_numpy(trainy).float().cuda()
        devX = torch.from_numpy(devX).float().cuda()
        devy = torch.from_numpy(devy).float().cuda()
        testX = torch.from_numpy(testX).float().cuda()
        testY = torch.from_numpy(testy).float().cuda()

        return trainX, trainy, devX, devy, testX, testy

    def run(self):
        self.nepoch = 0
        bestpr = -1
        early_stop_count = 0
        r = np.arange(1, 6)
        stop_train = False

        # Preparing data
        trainX, trainy, devX, devy, testX, testy = self.prepare_data(
            self.train['X'], self.train['y'],
            self.valid['X'], self.valid['y'],
            self.test['X'], self.test['y'])

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(trainX, trainy, nepoches=50)
            yhat = np.dot(self.predict_proba(devX), r)
            pr = spearmanr(yhat, self.devscores)[0]
            pr = 0 if pr != pr else pr  # if NaN bc std=0
            
            # early stop on Pearson
            if pr > bestpr:
                bestpr = pr
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel

        yhat = np.dot(self.predict_proba(testX), r)

        return bestpr, yhat

    def trainepoch(self, X, y, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
                Xbatch = X[idx]
                ybatch = y[idx]
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                if len(probas) == 0:
                    probas = self.model(Xbatch).data.cpu().numpy()
                else:
                    probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
        return probas

# Rickard: This is for full finetuning
class RelatednessPytorchFull(object):
    # Can be used for SICK-Relatedness, and STS14
    def __init__(self, train, valid, test, devscores, config, backbone, batcher, params):
        # fix seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        torch.cuda.manual_seed(config['seed'])

        self.batcher = batcher
        self.params = params

        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        self.inputdim = backbone.config.hidden_size*2 #train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.l2reg = 0.
        self.batch_size = 64
        self.maxepoch = 1 #
        self.early_stop = True

        self.backbone = backbone
        self.model = nn.Sequential(
            nn.Linear(self.inputdim, self.nclasses),
            nn.Softmax(dim=-1),
        )
        self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        #self.optimizer = optim.Adam(list(self.model.parameters())+list(self.backbone.parameters()),
        #                            weight_decay=self.l2reg)

        # Learning rate is important, depends on how many parameters are updated
        self.optimizer = optim.Adam(list(self.model.parameters())+list(self.backbone.parameters()), lr=0.0001) #+list(self.backbone.parameters()))

    def prepare_data(self, trainXA, trainXB, trainy, devXA, devXB, devy, testXA, testXB, testy):
        # Transform probs to log-probs for KL-divergence
        #print(trainXA) # actual subwords
        #print(trainXA)
        trainXA = trainXA # torch.from_numpy(trainXA).cuda() # torch.from_numpy(trainXA).float().cuda()
        trainXB = trainXB # torch.from_numpy(trainXB).cuda() # torch.from_numpy(trainXB).float().cuda()
        trainy = torch.from_numpy(trainy).float().cuda()
        devXA = devXA #  torch.from_numpy(devXA).cuda()  #torch.from_numpy(devXA).float().cuda()
        devXB = devXB # torch.from_numpy(devXB).cuda() #torch.from_numpy(devXB).float().cuda()
        devy = torch.from_numpy(devy).float().cuda()
        testXA = testXA #torch.from_numpy(testXA).cuda() #torch.from_numpy(testXA).float().cuda()
        testXB = testXB #torch.from_numpy(testXB).cuda() #torch.from_numpy(testXB).float().cuda()
        testy = testy #torch.from_numpy(testy).float().cuda()

        return trainXA, trainXB, trainy, devXA, devXB, devy, testXA, testXB, testy
    
    def run(self):
        self.nepoch = 0
        bestpr = -1
        early_stop_count = 0
        r = np.arange(1, 6)
        stop_train = False

        # Preparing data
        # trainX, trainy, devX, devy, testX, testy = self.prepare_data(
        #     self.train['X'], self.train['y'],
        #     self.valid['X'], self.valid['y'],
        #     self.test['X'], self.test['y'])
        
        trainXA, trainXB, trainy, devXA, devXB, devy, testXA, testXB, testy = self.prepare_data(
            self.train['X_A'], self.train['X_B'], self.train['y'],
            self.valid['X_A'], self.valid['X_B'], self.valid['y'],
            self.test['X_A'], self.test['X_B'], self.test['y']
        )

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(trainXA, trainXB, trainy, nepoches=2) #50)
            yhat = np.dot(self.predict_proba(devXA, devXB), r)
            pr = spearmanr(yhat, self.devscores)[0]
            print("[!] epoch: {}, spearmanr: {}".format(self.nepoch, pr))
            pr = 0 if pr != pr else pr  # if NaN bc std=0
            # early stop on Pearson

            if pr > bestpr:
                bestpr = pr
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        
        probas = self.predict_proba(testXA, testXB)
        #print("[!] probas", probas.shape, probas)
        yhat = np.dot(probas, r)

        return bestpr, yhat

    def trainepoch(self, XA, XB, y, nepoches=1):
        self.model.train()
        self.backbone.train()
        #for _ in range(self.nepoch, self.nepoch + nepoches):

        with tqdm(total=nepoches, desc="Training Progress") as pbar:
            for epoch in range(self.nepoch, self.nepoch + nepoches):
                #logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
                #print(_)
                permutation = np.random.permutation(len(XA))
                all_costs = []
                for i in range(0, len(XA), self.batch_size):
                    # forward
                    idx = torch.from_numpy(permutation[i:i + self.batch_size]).long() # .cuda()
                    XAbatch = [ XA[_i] for _i in idx ] # XA[idx]
                    XBbatch = [ XB[_i] for _i in idx ] # XB[idx]
                    ybatch = y[idx]
                    #print(ybatch)
                    #[[0.0000, 0.2000, 0.8000, 0.0000, 0.0000],                                                                                                                 
                    # [0.0000, 0.4000, 0.6000, 0.0000, 0.0000],

                    #####
                    ##### trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
                    ##### output from backbone
                    XAbatch, XBbatch = self.batcher(self.params, XAbatch), self.batcher(self.params, XBbatch)
                    outputA, outputB = self.backbone(XAbatch), self.backbone(XBbatch)
                    #print("SSSSS", outputA.shape, outputA.shape) # batch, dim
                    # torch.Size([64, 768]) torch.Size([64, 768])
                    Xbatch = torch.cat((torch.abs(outputA-outputB), outputA*outputB), dim=-1)
                    output = self.model(Xbatch)
                    # loss
                    loss = self.loss_fn(output, ybatch)
                    all_costs.append(loss.item())
                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Update parameters
                    self.optimizer.step()

                    # Update the progress bar
                    # Here, you can add any information you want to display in the progress bar
                    pbar.set_postfix({"Loss": loss.item(), "Epoch": epoch, "Batch": i})

                pbar.update(1)

        self.nepoch += nepoches

    def predict_proba(self, devXA, devXB):
        self.model.eval()
        self.backbone.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devXA), self.batch_size):
                XAbatch = devXA[i:i + self.batch_size]
                XBbatch = devXB[i:i + self.batch_size]

                XAbatch, XBbatch = self.batcher(self.params, XAbatch), self.batcher(self.params, XBbatch)
                outputA, outputB = self.backbone(XAbatch), self.backbone(XBbatch)
                Xbatch = torch.cat((torch.abs(outputA-outputB), outputA*outputB), dim=-1)
                output = self.model(Xbatch)
                preds = output.data.cpu().numpy()
                if len(probas) == 0:
                    probas = preds
                else:
                    probas = np.concatenate((probas, preds), axis=0)
        return probas
