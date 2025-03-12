import os
import torchvision
import torch
import numpy as np
import pickle
import math
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import models.clip as clip

def img_to_features(model, preprocess, args):
    img_dir = os.path.join(args.examples, 'images')
    imgset = torchvision.datasets.ImageFolder(img_dir)
    img_features = []
    
    with torch.no_grad():
        if args.model_clip == 'medclip':
            for image, _ in tqdm(imgset, desc='img2feat'):
                image = preprocess(images=image, return_tensors='pt')['pixel_values'].to(args.device)
                image_feature = model.encode_image(image)
                image_feature = image_feature.cpu().numpy()
                img_features.append(image_feature)

        img_features = np.concatenate(img_features, axis=0)
        with open(args.img_feat_save, 'wb') as f:
            pickle.dump(img_features, f)
    return img_features

def load_img_features(img_feat, detail=False):
    if detail:
        with open(img_feat, 'rb') as f:
            img_features = pickle.load(f)
    else:
        with open(img_feat, 'rb') as f:
            img_features = pickle.load(f)
    return img_features

def text_to_feature(all_words, model, preprocess, args, save=False):
    word_features = []

    with torch.no_grad():
        if args.model_clip == 'medclip':
            for word in tqdm(all_words, desc='text2feat'):
                text_inputs = preprocess(text=word, return_tensors='pt', padding=True)
                text_features = model.encode_text(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
                word_features.append(text_features.cpu().numpy())

    if save == True:
        word_features = np.concatenate(word_features, axis=0) 
        with open(args.word_feat_save, 'wb') as f:
            pickle.dump(word_features, f)
        
    else:
        word_features = np.array(word_features)

    return word_features

def load_word_features(word_feat):
    with open(word_feat, 'rb') as f:
        word_features = pickle.load(f)
    return word_features

def compute_concept_similarity(img_features, word_features, args, template_features=None, device='cuda', detail=False):
    concept_sim = []
    counter = 0
    img_features = torch.Tensor(img_features).to(device)
    word_features = torch.Tensor(word_features).to(device)
    if args.adaptive:
        template_features = torch.Tensor(template_features).to(device)

    if args.adaptive:
        for i in tqdm(range(len(img_features)), desc='adaptive'):
            img = img_features[i].reshape(1, -1)
            sim = cosine_similarity(img, word_features)
            template_sim = cosine_similarity(img, template_features)
            sim = sim - template_sim
            concept_sim.append([sim.cpu().numpy()])

        concept_sim = np.concatenate(concept_sim, axis=0)
        with open(args.concept_save, 'wb') as f:
            pickle.dump(concept_sim, f)
        
    else:
        for i in tqdm(range(len(img_features)), desc='no adaptive'):
            img = img_features[i].reshape(1, -1)
            sim = cosine_similarity(img, word_features)
            concept_sim.append([sim.cpu().numpy()])

        concept_sim = np.concatenate(concept_sim, axis=0)
        if detail: 
            with open(f'{args.concept_sim_root}/crop_sim_{args.num_example}_{counter}.pkl', 'wb') as f:
                pickle.dump(concept_sim, f)
        else:
            with open(args.concept_save, 'wb') as f:
                pickle.dump(concept_sim, f)

def concept_discovery(all_words, args, all_words_wotem=None, detail=False):
    
    img_weights = []
    if args.adaptive_percent is not None :
        with open(f'{args.examples}/num_example.pkl', 'rb') as f:
            num_example = pickle.load(f)
        c = 0
        idx = num_example[0]

    with open(args.concept_save, 'rb') as f:
        concept_sim = pickle.load(f)
    print(f'concept sim shape : {concept_sim.shape}')
    img_concept_weight = np.zeros(len(concept_sim[0]))

    for i in range(len(concept_sim)):
        if args.adaptive_percent is not None :
            if i%idx != 0 or i == 0:
                num_example_count = idx
            elif i == idx :
                num_example_count = idx #num_example[c]
                c += 1
                idx += num_example[c]
        else:
            num_example_count = args.num_example
        if i != 0 and i % num_example_count == 0:
            img_weights.append(img_concept_weight)
            img_concept_weight = np.zeros(len(concept_sim[0]))
        img_concept_weight += concept_sim[i]
    img_weights.append(img_concept_weight)
    
    concept_weight = []
    concept = []

    for i in range(len(img_weights)):
        max_sim = np.max(img_weights[i])
        threshold = max_sim * (args.alpha/100)
        img_concept_idx = np.where(img_weights[i] > threshold)[0]
        temp_weight = img_weights[i][img_concept_idx]
        concept_idx = np.argsort(img_weights[i][img_concept_idx])[::-1]
        concept_weight.append(temp_weight[concept_idx])

        concept_words = []
        if args.dataset == 1 or args.dataset == 20 or args.dataset == 365:
            for j in concept_idx:
                if args.template:
                    word = all_words_wotem[img_concept_idx[j]]
                else:
                    word = all_words[img_concept_idx[j]]
                concept_words.append(word)
            concept.append(concept_words)
        else:
            for j in concept_idx:
                if args.template:
                    word = all_words_wotem[img_concept_idx[j]]
                else:
                    word = all_words[img_concept_idx[j]]
                concept_words.append(word)
            concept.append(concept_words)
    pkl_save = f'{args.concept_sim_path}/pkl/{args.model_target}_{args.concept_set}{args.state}_{args.alpha}_{args.layer}.pkl'
    with open(pkl_save, 'wb') as f: 
        pickle.dump((concept,concept_weight), f)
    return concept, concept_weight, pkl_save