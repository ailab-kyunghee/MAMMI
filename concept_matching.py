import argparse
import torch
import os
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from utils.concept_matching_utils import img_to_features, load_img_features, text_to_feature, load_word_features, compute_concept_similarity, concept_discovery #, load_concept_sim
import json
import yaml as yaml
import models.clip as clip
import utils.med_metric_utils as metric
from sentence_transformers import SentenceTransformer

PROJECT = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description='Concept Matching')
    parser.add_argument('--feat_path', default='results/features', help='Path to save features(img & word)') #
    parser.add_argument('--concept_path', default='results/concept', help='Path to save concept') #
    parser.add_argument('--template_ver', default='base', help='base / locate / no')
    parser.add_argument('--adaptive', default='yes', help='yes / no')
    parser.add_argument('--model_target', default='densenet121', help='densenet121 / resnet50 / vit-b')
    
    parser.add_argument('--alpha', default=95, type=int, help='# of concept to select in img')
    parser.add_argument('--adaptive_percent', default=93, type=int, help='90~95')
    parser.add_argument('--dataset', default='nih-14')
    parser.add_argument('--concept_set', default='mimic_nouns')
    parser.add_argument('--layer', default='fc', help='fc / penultimate')
    parser.add_argument('--model_clip', default='medclip')
    return parser.parse_args()

def setting(args):
    # example root
    if args.dataset == 'nih-14':
        args.examples = f'{PROJECT}/results/examples/{args.model_target}/{args.dataset}_adap_{args.adaptive_percent}_examples/{args.layer}'
        
    if args.concept_set == 'mimic_nouns':
        args.concept_set_path = f'{PROJECT}/dataset/report/nouns.txt'

    args.concept_sim_path = os.path.join(args.concept_path, f'{args.dataset}_adap_{args.adaptive_percent}', args.model_clip)
    if not os.path.exists(args.concept_sim_path):
        os.makedirs(os.path.join(args.concept_sim_path, 'sim'), exist_ok=True)
        os.makedirs(os.path.join(args.concept_sim_path, 'pkl'), exist_ok=True)
    if args.template_ver == 'base' or args.template_ver == 'locate':
        args.state = f'_{args.template_ver}_tem'
        args.template = True
        if args.adaptive == 'yes':
            args.adaptive = True
        else:
            args.adaptive = False
    else:
        args.state = '_wotem'
        args.template = False
        args.adaptive = False

    # save feature (img / word)
    clip_feat_path = f'{args.feat_path}/{args.model_clip}'
    if not os.path.exists(clip_feat_path):
        os.makedirs(os.path.join(clip_feat_path, 'img'), exist_ok=True)
        os.makedirs(os.path.join(clip_feat_path, 'words'), exist_ok=True)
    args.img_feat_save = os.path.join(clip_feat_path, 'img', f'{args.model_target}_{args.dataset}_img_features_adap_{args.adaptive_percent}_{args.layer}.pkl')
    args.word_feat_save = os.path.join(clip_feat_path, 'words', f'{args.concept_set}_word_features{args.state}.pkl')

    if args.template_ver == 'base':
        args.base_template = ['A chest x-ray image of a']

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name, text=None):
    print(f'Load CLIP model : {model_name}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'medclip':
        preprocessor = MedCLIPProcessor()
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        model.cuda()
        return preprocessor, model
    
def main():
    args = parse_args()
    args.feat_path = os.path.join(PROJECT, args.feat_path)
    args.concept_path = os.path.join(PROJECT, args.concept_path)
    setting(args)

    ###### word to feature
    all_words = []

    # concept set : lower words
    with open(args.concept_set_path, 'r') as f:
        concept_words = f.read().split('\n')

    if args.template:
        all_words_wotem = []
        if args.template_ver == 'base':
            for word in concept_words:
                all_words.append(args.base_template[0]+' '+word)
                all_words_wotem.append(word)
    else:
        all_words_wotem = None
        all_words = concept_words

    preprocessor, model = load_model(args.model_clip, text=all_words)
    if not os.path.exists(args.word_feat_save):
        word_features = text_to_feature(all_words, model, preprocessor, args, save=True)
        template_features = text_to_feature(args.base_template, model, preprocessor, args)
    else:
        word_features = load_word_features(args.word_feat_save)
        template_features = text_to_feature(args.base_template, model, preprocessor, args)
    print(f'word feature shape : {word_features.shape}')
    if args.adaptive:
        args.state = args.state + '_adp'
    else:
        args.state = args.state + '_woadp'
    
    # image to feature
    if not os.path.exists(args.img_feat_save):
    # if True:
        img_features = img_to_features(model, preprocessor, args)
    else:
        img_features = load_img_features(args.img_feat_save)
    print(f'img feature shape : {img_features.shape}')
    args.concept_save = os.path.join(args.concept_sim_path, 'sim', f'{args.model_target}_{args.concept_set}_concept_sim{args.state}_adap_{args.adaptive_percent}_{args.layer}.pkl')
    
    # if not os.path.exists(args.concept_save):
    if True:
        compute_concept_similarity(img_features, word_features, args, template_features=template_features.squeeze())
    
    if True:   
        # concept_discovery(all_words, args, all_words_wotem=all_words_wotem, adaptive=adaptive, data=data, template=template)
        concept, concept_weight, target_pkl = concept_discovery(all_words, args, all_words_wotem=all_words_wotem)
    print(f'# of matched neuron: {len(concept)}')
    print(f'First neuron concept: {concept[0]}')

    # Metric
    print('* Start Metric *')
    save_dir, save_name = os.path.split(target_pkl)
    metric_dir, concept_dir = metric.make_dir(save_name, save_dir)

    if 'tem' in target_pkl.split('/')[-1].split('_'):
        tem = 'yes'
    else :
        tem = 'no'

    if args.layer == 'fc':
        gt, our_labels = metric.load_files(args.dataset, target_pkl, args.template_ver, tem)
        concept = [{'GT':g, 'Pred':l} for g,l in zip(gt, our_labels)]
        
        with open(concept_dir,'w') as f:
            json.dump(concept, f, indent=4)
    else :
        with open(concept_dir,'w') as f:
            json.dump(concept[0], f, indent=4)
        
        exit(0)

    results = []
    
    # Hit-rate
    ours_acc, _ = metric.ours_accuracy(our_labels, gt)
    acc_dict = {'Hit rate':round(ours_acc, 4)}
    results.append(acc_dict)
    print('Finish : Hit rate')
    
    # Similarity
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    clip_vit, _ = clip.load('ViT-B/16', device=args.device)
    med_clip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    med_clip.from_pretrained()
    similarity = metric.similarities(med_clip, clip_vit, mpnet_model, our_labels, gt, 200, args.device)
    results.append(similarity)
    print('Finish : Cosine Similarity')

    # F1-score
    pr = metric.pr(our_labels, gt)
    results.append(pr)
    print('Finish : Precision & Recall')

    results[0], results[1], results[2] = results[1], results[2], results[0]
    with open(metric_dir,'w') as f:
        json.dump(results, f, indent=4)
    print('score :', results)
    print('Finish-concept matching & metric')

if __name__ == '__main__':
    main()