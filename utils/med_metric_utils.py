import argparse
import pickle
import json
import os
import numpy as np
from tqdm import tqdm
#virtually move to parent directory
os.chdir("..")

import torch
import pandas as pd

from torch.nn.functional import cosine_similarity
# from sentence_transformers import SentenceTransformer

import clip ## clip-dissect
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

def make_dir(ours, save_dir):
    score_save_dir = save_dir.replace(os.path.split(save_dir)[1], 'score')
    # save_dir = os.path.join(save_dir, 'score')
    if not os.path.exists(score_save_dir):
        os.makedirs(score_save_dir, exist_ok=True)
    
    # args.match_concept_save_dir = args.save_dir.replace(os.path.split(args.save_dir)[1], 'match_concept')
    match_concept_save_dir = save_dir.replace(os.path.split(save_dir)[1], 'match_concept')
    if not os.path.exists(match_concept_save_dir):
        os.makedirs(match_concept_save_dir, exist_ok=True)

    # target = ours.split('/')[-1].split('.')[0]
    target = ours.replace('pkl', 'json')
    
    score_save = f'{score_save_dir}/{target}'
    concept_save = f'{match_concept_save_dir}/{target}'
    
    return score_save, concept_save

def load_files(gt, ours, tem_version='base', tem='yes'):
    if 'categories_places365' == gt.split('/')[-1].split('.')[0]:
        with open (gt, 'r') as f:
            gt = (f.read().split('\n'))
        gt = [label.split('/')[-1].split(' ')[0].lower() for label in gt]
    elif gt == 'nih-14':
        util_dir = os.path.dirname(os.path.realpath(__file__))
        gt_path = f'{util_dir}/nih_labels.txt'
        with open(gt_path, 'r') as f: 
            gt = (f.read()).split('\n')
    else :
        with open(gt, 'r') as f: 
            gt = (f.read()).split('\n')
    if type(ours) == list:
        with open(ours[0], 'rb') as f: 
            covid_labels = pickle.load(f)
        with open(ours[1], 'rb') as f: 
            nih_labels = pickle.load(f)

        assert len(covid_labels) == len(nih_labels), 'check the labels file'
        our_labels = []
        for i in range(len(covid_labels)):
            covid = covid_labels[i]
            nih = nih_labels[i]
            if covid[3] > nih[3]:
                concept = covid[2]
                if concept.lower() == 'no class':
                    concept = 'none'
                elif concept.lower() == 'infiltrate':
                    concept = 'infiltration'
                our_labels.append([concept])
            else:
                concept = nih[2]
                if concept.lower() == 'no class':
                    concept = 'none'
                elif concept.lower() == 'infiltrate':
                    concept = 'infiltration'
                our_labels.append([concept])
    else:
        with open(ours, 'rb') as f: 
            our_labels = pickle.load(f)
            if len(our_labels) == 2:
                our_labels = our_labels[0]
            elif len(our_labels[0]) == 4:
                our_labels = [[label[2]] for label in our_labels]

    if tem == 'yes':
        results = []
        if tem_version == 'base':
            template = 'A chest x-ray image of a '
            for our in our_labels:
                temp = []
                for o in our:
                    wotem = o.replace(template, '')
                    temp.append(wotem)
                results.append(temp)
            our_labels = results
        elif tem_version == 'location':
            for dic in our_labels:
                temp = []
                for our in dic:
                    wotem = our['concept']
                    temp.append(wotem)
                results.append(temp)
            our_labels = results

    # if mode == 'cd':
    #     our_labels_c = []
    #     top_n = cd_num
    #     for temp_our in our_labels:
    #         new = temp_our[:top_n]
    #         our_labels_c.append(new)
    #     our_labels = our_labels_c

    our_lower = []
    gt_lower = []
    for o in our_labels:
        if o == []:
            our_lower.append(['none'])
        else:
            our_l = [l.lower() for l in o]
            our_lower.append(our_l)
    for idx in range(len(gt)):
        g = gt[idx].split(', ')
        gt_l = [gl.lower() for gl in g]
        gt_lower.append(gt_l)

    return gt_lower, our_lower

def ours_accuracy(ours, gt):
    new = []
    check = [False] * len(ours)
    for i in range(len(ours)):
        if check[i] == False:
            for our in ours[i]:
                if check[i] == False:
                    if our in gt[i]:
                        check[i] = True
                        new.append([i, ours[i], gt[i]])

    ours_acc = float(sum(check) / len(ours)) 
    return ours_acc, new

def compute_sim(preds, gt, med_clip, clip_vit, mpnet_model, device='cuda', batch_size=200):
    #### med clip
    # tokenize : clip_model.text_model.tokenizer(preds)
    # decoding : clip_model.text_model.tokenizer._decode([101])
    med_clip.to(device)
    # pred_tokens_ = clip_model.text_model.tokenizer(preds)
    # gt_tokens_ = clip_model.text_model.tokenizer(gt)

    preprocessor = MedCLIPProcessor()
    pred_inputs = preprocessor(text=preds, return_tensors='pt', padding=True)
    gt_inputs = preprocessor(text=gt, return_tensors='pt', padding=True)
    pred_tokens = pred_inputs['input_ids']
    gt_tokens = gt_inputs['input_ids']
    # pred_tokens = clip.tokenize(preds).to(device)
    # gt_tokens = clip.tokenize(gt).to(device)
    cos_sim_list = []
    for i in range(len(pred_tokens)):
        p = med_clip.encode_text(pred_tokens[i:i+1])
        for j in range(len(gt_tokens)):
            g = med_clip.encode_text(gt_tokens[j:j+1])
            cos_sim = cosine_similarity(p, g)
            if float(cos_sim) > 1:
                cos_sim = 1.0
            sim_dict = ({'pred':preds[i], 'gt':gt[j], 'sim':float(cos_sim)})
            cos_sim_list.append(sim_dict)
            del cos_sim
            del g
        del p
    del pred_tokens
    del gt_tokens
    torch.cuda.empty_cache()

    #### vit clip
    pred_tokens = pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = gt_tokens = clip.tokenize(gt).to(device)
    vit_sim_list = []
    for i in range(len(pred_tokens)):
        p = clip_vit.encode_text(pred_tokens[i:i+1])
        for j in range(len(gt_tokens)):
            g = clip_vit.encode_text(gt_tokens[j:j+1])
            cos_sim = cosine_similarity(p, g)
            if float(cos_sim) > 1:
                cos_sim = 1.0
            
            sim_dict = ({'pred':preds[i], 'gt':gt[j], 'sim':float(cos_sim)})
            vit_sim_list.append(sim_dict)
            del cos_sim
            del g
        del p
    del pred_tokens
    del gt_tokens
    torch.cuda.empty_cache()

    #### mpnet
    mp_sim_list = []
    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    for i in range(len(pred_embeds)):
        p = pred_embeds[i:i+1]
        mp_sim = np.sum(p*gt_embeds, axis=1)
        cos_sim_mpnet = float(np.mean(mp_sim))
        mp_sim_list.append(cos_sim_mpnet)
    return cos_sim_list, vit_sim_list, mp_sim_list

def similarities(med_clip, clip_vit, mpnet_model, our_labels, gt, batch_size='200', device='cpu'):
    med_clip_sim_list = []
    vit_clip_sim_list = []
    mp_sim_list = []
    for id in tqdm(range(len(gt)), desc='(Similarity) # of neuron: '):
        torch.cuda.empty_cache()
        n_id = 'neuron_id:{}'.format(id)

        med_clip_cos, vit_clip_cos, mpnet_cos = compute_sim(our_labels[id], gt[id],
                                        med_clip, clip_vit, mpnet_model, device, batch_size)
        
        # med_CLIP cosine
        temp_cos = []
        for s in med_clip_cos:
            sim = float(s['sim'])
            temp_cos.append(sim)
        med_clip_sim_list.append(np.mean(temp_cos))

        # vit_CLIP cosine
        temp_cos = []
        for s in vit_clip_cos:
            sim = float(s['sim'])
            temp_cos.append(sim)
        vit_clip_sim_list.append(np.mean(temp_cos))

        # MPNet cosine
        mp_sim_list.append(np.mean(mpnet_cos))

    med_clip_result = {'Cosine':round(np.mean(med_clip_sim_list), 4), 
                   'St_error':round(np.std(med_clip_sim_list)/(len(gt)**(1/2)), 4)}
    
    vit_clip_result = {'Cosine':round(np.mean(vit_clip_sim_list), 4), 
                   'St_error':round(np.std(vit_clip_sim_list)/(len(gt)**(1/2)), 4)}
        
    mpnet_result = {'Cosine':round(np.mean(mp_sim_list), 4), 
                    'St_error':round(np.std(mp_sim_list)/(len(gt)**(1/2)), 4)}
    
    result_dict = {'Med CLIP':med_clip_result, 
                   'ViT CLIP':vit_clip_result,
                   'MPNet':mpnet_result}
        
    return result_dict

def intersection(a, b):
    aa, bb = set(a), set(b)
    item = list(aa&bb)
    intersect = len(item)
    return item, intersect

def compute_pr(gt, our):
    item, intersect = intersection(gt, our)
    precision = intersect/len(our)
    recall = intersect/len(gt)
    return precision, recall, intersect

def pr(our_labels, gt):
    p_list = []
    r_list = []
    f1_list = []
    for i in range(len(gt)):
        g = gt[i]
        our = our_labels[i]
        pre, re, intersect = compute_pr(g, our)
        if pre + re == 0:
            f1 = 0.0
        else:
            f1 = 2 * (pre * re) / (pre + re)
        p_list.append(pre)
        r_list.append(re)
        f1_list.append(f1)
    precision_mean, precision_std = round(np.mean(p_list),4), round(np.std(p_list),4)
    recall_mean, recall_std = round(np.mean(r_list),4), round(np.std(r_list),4)
    f1_mean, f1_std = round(np.mean(f1_list),4), round(np.std(f1_list),4)
    precision = {'Precision':precision_mean, 'St_error':round(precision_std/(len(gt)**(1/2)),4)}
    recall = {'Recall':recall_mean, 'St_error':round(recall_std/(len(gt)**(1/2)),4)}
    f1_score = {'F1_score':f1_mean, 'St_error':round(f1_std/(len(gt)**(1/2)),4)}

    result_dict = {'F1_score':f1_score, 'Recall':recall, 'Precision':precision}

    return result_dict