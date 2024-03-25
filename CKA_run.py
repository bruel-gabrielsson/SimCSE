from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random
import json
import numpy as np
#import tensorboard_logger as tb_logger
# from torch.utils.tensorboard import SummaryWriter 
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn.functional as F

from collections import defaultdict

#from util import TwoCropTransform, AverageMeter
#from util import adjust_learning_rate, warmup_learning_rate
#from util import set_optimizer, save_model
#from util import MyTransforms, plot_some

#from main_uni import set_loader, set_model

from PIL import Image

from CKA import linear_CKA, kernel_CKA

import matplotlib.pyplot as plt

import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/downstream/STS/STSBenchmark'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval 


# STSBenchmarkEval

'''
vectors = np.array([[0,0,1], [0,1,0], [1,0,0], [1,1,1]])
metadata = ['001', '010', '100', '111']  # labels
writer = SummaryWriter()
writer.add_embedding(vectors, metadata)
writer.close()
'''

CUDA_VISIBLE_DEVICES="1"


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def main():
    print("[!] CKA CKA CKA CKA CKA CKA CKA")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument("--name", type=str, 
            default='no_name', 
            help="name of the experiment")

    
    
    args = parser.parse_args()

    '''
    model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
    '''
    
    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path) # , dropout_only_layer=-3) # always -3
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # RICKARD: Why wasn't this here before?

    #params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                        'tenacity': 3, 'epoch_size': 2}
    params = Dict2Class(params)
                                         
    ##### DATASET #####

    sTSBenchmarkEval = senteval.STSBenchmarkEval(PATH_TO_DATA)

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Get raw embeddings
        with torch.no_grad():
            # MODEL DONE HERE
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
            

        # Apply different poolers
        pooler_final_result = None
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            pooler_final_result = pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            pooler_final_result = last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            pooler_final_result = ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            pooler_final_result = pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            pooler_final_result = pooled_result.cpu()
        else:
            raise NotImplementedError

        #print(len(hidden_states)) # 13
        #embeddings, hiddens = hidden_states
        #print(pooler_final_result.shape)
        #print(embeddings.shape, hiddens.shape)

        #return pooler_final_result
        return hidden_states
    ######
    
    TYPE="kernel"
    if TYPE == "linear":
        CKA_run = linear_CKA
    elif TYPE == "kernel":
        CKA_run = kernel_CKA
    else:
        raise ValueError("TYPE must be linear or kernel")

    name="{}_{}".format(args.name, TYPE)
    

    def model_to_dict_acts(model):
        model.eval()
        model_dict = defaultdict(lambda: [])

        with torch.no_grad():
            all_embeddings = []

            '''
            We take STS-B pairs with a score higher than 4 as ppos and all STS-B sentences as pdata.
            '''
            p_data = []
            p_pos1 = []
            p_pos2 = []
            for dataset in sTSBenchmarkEval.datasets:
                print("dataset: ",dataset)
                if dataset != "dev":
                    continue
                
                sys_scores = []
                input1, input2, gs_scores = sTSBenchmarkEval.data[dataset]
                #params.batch_size=1
                for ii in range(0, len(gs_scores)//8, params.batch_size): # params.batch_size): #  len(gs_scores) //4
                        
                    batch1 = input1[ii:ii + params.batch_size]
                    batch2 = input2[ii:ii + params.batch_size]

                    # we assume get_batch already throws out the faulty ones
                    if len(batch1) == len(batch2) and len(batch1) > 0 and len(batch1)==params.batch_size:
                        enc1 = batcher(params, batch1)
                        enc2 = batcher(params, batch2)
                        
                        for _i in range(len(enc1)):
                            #print(enc1[_i].shape)
                            model_dict[_i].append(enc1[_i].view(-1, 768))
                            model_dict[_i].append(enc2[_i].view(-1, 768))



        #     end = time.time()
        #     for idx, (images, labels) in enumerate(test_loader):
        #         images = images.float().cuda()
        #         labels = labels.cuda()

        #         encoder = model.encoder.module

        #         x = images
        #         x = F.relu(encoder.bn1(encoder.conv1(x)))
        #         model_dict["layer0"].append(x)
        #         x = encoder.layer1(x)
        #         model_dict["layer1"].append(x)
        #         x = encoder.layer2(x)
        #         model_dict["layer2"].append(x)
        #         x = encoder.layer3(x)
        #         model_dict["layer3"].append(x)
        #         x = encoder.layer4(x)
        #         model_dict["layer4"].append(x)
        #         x = encoder.avgpool(x)
        #         x = torch.flatten(x, 1)
        #         model_dict["layer5"].append(x)
        #         x = model.head(x)
        #         model_dict["layer6"].append(x)

        for key in model_dict.keys():
            model_dict[key] = torch.flatten(torch.cat(model_dict[key], dim=0),1).detach().cpu().numpy()
            print(key, model_dict[key].shape)
        
        return model_dict

    dict1 = model_to_dict_acts(model)
    dict2 = model_to_dict_acts(model)

    comparision_matrix = np.zeros([len(dict1), len(dict2)])
    keys1, keys2 = list(dict1.keys()), list(dict2.keys())
    for i in range(len(dict1)):
        for j in range(len(dict2)):
            print(i,j)
            comparision_matrix[i,j] = CKA_run(dict1[keys1[i]], dict2[keys2[j]])

    np.savetxt('comparision_matrix_{}.txt'.format(name), comparision_matrix, delimiter=',')
    print("saved comparision_matrix_{}.txt".format(name))
    print(comparision_matrix)
    plt.imshow(comparision_matrix, vmin=0.0, vmax=1.0) # , interpolation='nearest')
    plt.colorbar()
    plt.savefig("comparision_matrix_{}.png".format(name))

    # model2 
    # dict2 = model_to_dict_acts(model2, test_loader)

    # all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
    # all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()
    # all_images = torch.cat(all_images, dim=0).detach().cpu()
    # print(all_images.shape)
    # print(all_embeddings.shape)
    # print(all_labels.shape)

    # one_square_size = int(np.ceil(np.sqrt(len(all_embeddings))))
    # master_width = 100 * one_square_size
    # master_height = 100 * one_square_size
    # spriteimage = Image.new(
    #     mode='RGBA',
    #     size=(master_width, master_height),
    #     color=(0,0,0,0) # fully transparent
    # )

    # for count, image in enumerate(images_pil):
    #     div, mod = divmod(count, one_square_size)
    #     h_loc = 100 * div
    #     w_loc = 100 * mod
    #     spriteimage.paste(image, (w_loc, h_loc))
    # spriteimage.convert(“RGB”).save(f’{LOG_DIR}/embeddings/sprite.jpg’, transparency=0)
    

if __name__ == '__main__':
    main()

