# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging

import torch

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine
from senteval.sick import SICKEval


class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            #print(sent1)

            # sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            # sent2 = np.array([s.split() for s in sent2])[not_empty_idx]

            _sent1, _sent2 = [], []
            for _i in range(len(sent1)):
                if not_empty_idx[_i]:
                    _sent1.append(sent1[_i].split())
                    _sent2.append(sent2[_i].split())
            sent1,sent2 = _sent1,_sent2    

            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
            
            #print(sys_scores,gs_scores)
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results

class STSEvalAlignUni(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)


    def alignment(self,x,y):
        x,y = torch.nn.functional.normalize(x, dim=1, p=2).cpu().detach(), torch.nn.functional.normalize(y, dim=1, p=2).cpu().detach()
        return (x-y).pow(2).sum(-1).mean()


    def uniformity(self,x):
        x = torch.nn.functional.normalize(x, dim=1, p=2).cpu().detach()
        return (torch.pdist(x,p=2).mul(-2).exp().mean().log())

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []

        all_embeddings = []

        '''
        We take STS-B pairs with a score higher than 4 as ppos and all STS-B sentences as pdata.
        '''
        p_data = []
        p_pos1 = []
        p_pos2 = []
        for dataset in self.datasets:
            print("dataset: ",dataset)
            

            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    all_embeddings.append(enc1)
                    all_embeddings.append(enc2)
                    
                    p_data.append(enc1)
                    p_data.append(enc2)

                    for _i in range(len(enc1)):
                        if gs_scores[_i] > 4:
                            #print(enc1[_i].shape)
                            p_pos1.append(enc1[_i].unsqueeze(0))
                            p_pos2.append(enc2[_i].unsqueeze(0))


            

            

        #     all_sys_scores.extend(sys_scores)
        #     all_gs_scores.extend(gs_scores)
        #     results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
        #                         'spearman': spearmanr(sys_scores, gs_scores),
        #                         'nsamples': len(sys_scores)}
        #     logging.debug('%s : pearson = %.4f, spearman = %.4f' %
        #                   (dataset, results[dataset]['pearson'][0],
        #                    results[dataset]['spearman'][0]))

        # weights = [results[dset]['nsamples'] for dset in results.keys()]
        # list_prs = np.array([results[dset]['pearson'][0] for
        #                     dset in results.keys()])
        # list_spr = np.array([results[dset]['spearman'][0] for
        #                     dset in results.keys()])

        # avg_pearson = np.average(list_prs)
        # avg_spearman = np.average(list_spr)
        # wavg_pearson = np.average(list_prs, weights=weights)
        # wavg_spearman = np.average(list_spr, weights=weights)
        # all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        # all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        # results['all'] = {'pearson': {'all': all_pearson[0],
        #                               'mean': avg_pearson,
        #                               'wmean': wavg_pearson},
        #                   'spearman': {'all': all_spearman[0],
        #                                'mean': avg_spearman,
        #                                'wmean': wavg_spearman}}
        # logging.debug('ALL : Pearson = %.4f, \
        #     Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        # logging.debug('ALL (weighted average) : Pearson = %.4f, \
        #     Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        # logging.debug('ALL (average) : Pearson = %.4f, \
        #     Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        p_pos1 = torch.cat(p_pos1, dim=0)
        p_pos2 = torch.cat(p_pos2, dim=0)
        p_data = torch.cat(p_data, dim=0)
        #print(p_pos1.shape, p_pos2.shape, p_data.shape)
        # torch.Size([1889, 768]) torch.Size([1889, 768]) torch.Size([11498, 768])

        alignment = self.alignment(p_pos1, p_pos2).item()
        uniformity = self.uniformity(p_data).item()

        print("Alignment: ", alignment)
        print("Uniformity: ", uniformity)

        results = {'alignment':  alignment, 'uniformity': uniformity}

        return results

    # def uniformity_and_alignment(x,y):
    #     lalign=(x-y).pow(2).sum(-1).mean()
    #     lunif=(torch.pdist(x,p=2).mul(-2).exp().mean().log() +
    #         torch.pdist(y,p=2).mul(-2).exp().mean().log())/2

    #     return lunif, lalign

class STSBenchmarkEvalAlignUni(STSEvalAlignUni):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

from sklearn.metrics import mean_squared_error
from senteval.tools.relatedness import RelatednessPytorchFull

class STSBenchmarkFulltrain(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])
                # print(sick_data) 
                # {'X_A': [['a', 'plane', 'is', 'taking', 'off', '.']], 'X_B': [['an', 'air', 'plane', 'is', 'taking', 'off', '.']], 'y': ['5.000']}
                # assert(False)

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher, backbone):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    #tokens = batcher(params, batch)
                    #print(tokens)
                    #assert(False)
                    sick_embed[key][txt_type].extend(batch)

                #print(sick_embed[key][txt_type]) # a list of list of lists?
                #sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        #trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        #devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        #testF = np.c_[np.abs(testA - testB), testA * testB]

        '''
        def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j, i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j, i] = np.floor(y) - y + 1
        return Y
        '''
        testY = self.encode_labels(self.sick_data['test']['y'])

        config = {'seed': self.seed, 'nclasses': 5}
        #backbone.train()
        clf = RelatednessPytorchFull(train={'X_A': trainA, 'X_B': trainB, 'y': trainY},
                                 valid={'X_A': devA, 'X_B': devB, 'y': devY},
                                 test={'X_A': testA, 'X_B': testB,'y': testY},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config, backbone=backbone, batcher=batcher, params=params)

        devspr, yhat = clf.run()

        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        print("[!] pr, sr: ", pr, sr)
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(yhat, self.sick_data['test']['y'])
        logging.debug('Dev : Spearman {0}'.format(devspr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                       for SICK Relatedness\n'.format(pr, sr, se))

        return {'devspearman': devspr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(devA), 'ntest': len(testA)}
    

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
