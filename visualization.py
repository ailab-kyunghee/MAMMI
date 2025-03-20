import argparse
import numpy as np
import os
import cv2
import pickle as pkl
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import cosine_similarity
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

import utils.util as util
from dataset.dataloader_med import Augmentation, ChestX_ray14, ChestX_ray14_det, ChestX_ray14_bbox
PROJECT = os.path.dirname(os.path.realpath(__file__))

def detail_setting(args):
    NIH_DET_PATH = 'YOUR_PATH' # f'{PROJECT}/dataset/nih-det_box/{args.split}_'
    if args.dataset == 'nih-det':
        args.dataset_path = NIH_DET_PATH
        args.split_path = f'{PROJECT}/dataset/det_split/ChestX_Det_{args.split}.json'

    class_path = f'{PROJECT}/dataset/nih_split/nih_labels.txt'
    with open(class_path, 'r') as f: 
        args.class_name = (f.read()).split('\n')

    # Threshold
    args.thrs_path = f'{PROJECT}/results/model_perform/{args.model}/results/Threshold.csv'

    # check point
    if args.model == 'densenet121':
        args.check_path = f'{PROJECT}/pretrained/target_model/{args.model}_CXR_0.3M_mocov2.pth'

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_threshold(thrs_path):
    Eval = pd.read_csv(thrs_path)
    thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Pleural thickening"].index[0]],
            Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]
    return thrs

def main(args):
    detail_setting(args)

    ##### Multi-class classification threshold
    thrs = load_threshold(args.thrs_path)

    ##### Dataset
    transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")
    no_normalize = Augmentation(normalize="none").get_augmentation("full_224", "val")
    if args.dataset == 'nih':
        if args.split == 'bbox':
            p_data = ChestX_ray14_bbox(args.dataset_path, args.split_path, augment=transform, no_normalize_aug=no_normalize, num_class=14)
        else:
            p_data = ChestX_ray14(args.dataset_path, args.split_path, augment=transform, num_class=14)
        
    elif args.dataset == 'nih-det':
        p_data = ChestX_ray14_det(args.dataset_path, args.split_path, augment=transform, no_normalize_aug=no_normalize, num_class=14)

    sampler = torch.utils.data.SequentialSampler(p_data)
    loader = torch.utils.data.DataLoader(
        p_data, sampler=sampler,
        batch_size=1, #args.batch_size,
        num_workers=4, #args.num_workers,
        pin_memory=True, #args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    ##### Load model
    model = util.load_model(args.check_path, args.model, args.device)
    
    ##### heatmap color
    cmaps = [
        util.get_alpha_cmap((54, 197, 240)),##blue
        util.get_alpha_cmap((210, 40, 95)),##red
        util.get_alpha_cmap((236, 178, 46)),##yellow
        util.get_alpha_cmap((15, 157, 88)),##green
        util.get_alpha_cmap((84, 25, 85)),##purple
        util.get_alpha_cmap((255, 0, 0))##real red
    ]
    to_pil = ToPILImage()
    
    total_results = f'{args.heatmap_save_root}/total_results.json'
    detail_info = {} 

    if args.dataset == 'nih-det':
        disease_color_map = {'Atelectasis': 'Red',
                            'Calcification': 'Green',
                            'Cardiomegaly': 'Blue',
                            'Consolidation': 'Yellow',
                            'Diffuse Nodule': 'Magenta',
                            'Effusion': 'Cyan',
                            'Emphysema': 'Dark Red',
                            'Fibrosis': 'Dark Green',
                            'Fracture': 'Dark Blue',
                            'Mass': 'Olive',
                            'Nodule': 'Purple',
                            'Pleural Thickening': 'Teal',
                            'Pneumothorax': 'Gray'}
    elif args.dataset == 'nih':
        disease_color_map = {'Atelectasis': 'Red',
                            'Infiltrate': 'Green',
                            'Cardiomegaly': 'Blue',
                            'Consolidation': 'Yellow',
                            'Pneumonia': 'Magenta',
                            'Effusion': 'Cyan',
                            'Emphysema': 'Dark Red',
                            'Fibrosis': 'Dark Green',
                            'Edema': 'Dark Blue',
                            'Mass': 'Olive',
                            'Nodule': 'Purple',
                            'Pleural Thickening': 'Teal',
                            'Pneumothorax': 'Gray',
                            'Hernia': 'Orange'}
    
    for j, (img, _, _, label, _, img_name, no_norm_img, no_norm_box, _) in enumerate(tqdm(loader)):
        img_name = img_name[0].split('.')[0]
        predict = model(img.to(args.device))
        predict = predict.sigmoid()[0].cpu().detach().numpy()
        
        pred_idx = np.where(predict >= thrs)[0]
        pred_class = [args.class_name[i] for i in pred_idx]

        if pred_class != []:
            save_img_base = f'{args.heatmap_save_root}/img/{int(img_name):06d}'
            os.makedirs(save_img_base, exist_ok=True)

            # origin image save : original, bbox
            image_crop = to_pil(no_norm_img.squeeze())
            image_crop.save(f'{save_img_base}/crop_original.jpg')

            box_crop = to_pil(no_norm_box.squeeze())
            box_crop.save(f'{save_img_base}/crop_box.jpg')

            # each image information
            img_info = {} 
            gt_label = []
            for l in label:
                disease = l[0]
                color_map = disease_color_map[disease]
                gt_label.append((disease,color_map))
            img_info['GT'] = gt_label

            # feature map
            feature_maps = model.extract_feature_map_4(img.cuda())
            feature_maps = feature_maps[0].cpu().detach().numpy()
            feature_maps = feature_maps.transpose(1, 2, 0)

            for p_idx, pred in enumerate(zip(pred_class, pred_idx)):
                pred_info = {}

                pred = list(pred)
                pred_c = pred[0]
                pred_i = pred[1]

                pred_info['Prediction'] = (pred[0], pred[1].tolist(), predict[pred_i].tolist())
                
                ###### Sample attribution ######
                util.show(img[0])
                sample_shap = model._compute_taylor_scores(img.cuda(), [pred_i])
                sample_shap = sample_shap[0][0][0,:,0,0]
                sample_shap = sample_shap.cpu().detach().numpy()
                most_important_concepts = np.argsort(sample_shap)[::-1][:args.num_top_neuron]
                for i, c_id in enumerate(most_important_concepts):
                    cmap = cmaps[i]
                    heatmap = feature_maps[:, :, c_id]

                    sigma = np.percentile(feature_maps[:,:,c_id].flatten(), args.percentile)
                    heatmap = heatmap * np.array(heatmap > sigma, np.float32)

                    heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                    util.show(heatmap, cmap=cmap, alpha=0.9)
                save_sample_att = f'{save_img_base}/sample_attribute_{pred_c}.jpg'
                plt.savefig(save_sample_att, bbox_inches='tight', pad_inches=0)
                plt.clf()

                ###### Sample overall attribution ######
                util.show(img[0])
                sample_overall_heatmap = np.zeros((224, 224))
                for i, c_id in enumerate(most_important_concepts):
                    heatmap = feature_maps[:, :, c_id]
                    heatmap = cv2.resize(heatmap[:, :, None], (224, 224))
                    weight = sample_shap[c_id] / np.sum(sample_shap[most_important_concepts])
                    #
                    sigma = np.percentile(heatmap.flatten(), args.percentile_overall)
                    heatmap = heatmap * np.array(heatmap > sigma, np.float32)
                    #
                    sample_overall_heatmap += heatmap * weight

                util.show(sample_overall_heatmap, cmap='Reds', alpha=0.5)
                save_sample_overall_att = f'{save_img_base}/sample_overall_attribute_{pred_c}.jpg'
                plt.savefig(save_sample_overall_att, bbox_inches='tight', pad_inches=0)
                plt.clf()

                pred_info['Sample'] = {'most important neurons':most_important_concepts.tolist(), 
                                       'path': {'sample': save_sample_att,
                                                'sample_overall': save_sample_overall_att}}
                
                img_info[f'pred_{p_idx}'] = pred_info
                
            detail_info[img_name] = img_info
            with open(total_results, 'w') as f:
                json.dump(detail_info, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize heatmap with box')
    parser.add_argument('--dataset', default='nih-det', help='nih / nih-det')
    parser.add_argument('--split', default='test', help='nih-det:test,train / nih:bbox')
    parser.add_argument('--model', default='densenet121', help='densenet121, resnet50, vit-b/16')
    parser.add_argument('--heatmap_save_root', default='results/visualization')
    parser.add_argument('--num_top_neuron', type=int, default=1)
    parser.add_argument('--percentile', type=int, default=70, help='activation map')
    parser.add_argument('--percentile_overall', type=int, default=90, help='overall activation map')
    args = parser.parse_args()
    
    args.heatmap_save_root = f'{PROJECT}/{args.heatmap_save_root}/{args.dataset}_{args.split}'
    main(args)