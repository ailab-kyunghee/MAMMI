import cv2
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd

PROJECT = os.path.dirname(os.path.realpath(__file__))

def apply_color_palette(box_mask, sym, dataset):
    if dataset == 'nih-det':
        disease_color_map = {
            'Atelectasis': (255, 0, 0),         # Red 
            'Calcification': (0, 255, 0),       # Green
            'Cardiomegaly': (0, 0, 255),        # Blue 
            'Consolidation': (255, 255, 0),     # Yellow
            'Diffuse Nodule': (255, 0, 255),    # Magenta
            'Effusion': (0, 255, 255),          # Cyan 
            'Emphysema': (128, 0, 0),           # Dark Red
            'Fibrosis': (0, 128, 0),            # Dark Green
            'Fracture': (0, 0, 128),            # Dark Blue
            'Mass': (128, 128, 0),              # Olive
            'Nodule': (128, 0, 128),            # Purple
            'Pleural Thickening': (0, 128, 128), # Teal
            'Pneumothorax': (128, 128, 128)     # Gray
        }
    elif dataset == 'nih':
        disease_color_map = {
            'Atelectasis': (255, 0, 0),         # Red 
            'Infiltrate': (0, 255, 0),        # Green
            'Cardiomegaly': (0, 0, 255),        # Blue 
            'Consolidation': (255, 255, 0),     # Yellow
            'Pneumonia': (255, 0, 255),         # Magenta
            'Effusion': (0, 255, 255),          # Cyan 
            'Emphysema': (128, 0, 0),           # Dark Red
            'Fibrosis': (0, 128, 0),            # Dark Green
            'Edema': (0, 0, 128),               # Dark Blue
            'Mass': (128, 128, 0),              # Olive
            'Nodule': (128, 0, 128),            # Purple
            'Pleural Thickening': (0, 128, 128), # Teal
            'Pneumothorax': (128, 128, 128),     # Gray
            'Hernia': (255, 128, 0)             # Orange
        }
    
    palette = [(0, 0, 0)]
    for s_i in range(len(sym)):
        palette.append(disease_color_map.get(sym[s_i], (128, 128, 128)))
    
    palette = np.array(palette)
    colored_mask = palette[box_mask.astype(int)]

    return colored_mask.astype(np.uint8)

def masking(dataset, box_mask, entry, sym):
    if dataset == 'nih-det':
        for i, box in enumerate(entry.get("boxes", [])):
            x1, y1, x2, y2 = box
            box_mask[y1:y2, x1:x2] = i + 1  
    elif dataset == 'nih':
        x, y, w, h = int(entry['bbox_x']), int(entry['bbox_y']), int(entry['bbox_w']), int(entry['bbox_h'])
        x1, y1, x2, y2 = x, y, x+w, y+h
        box_mask[y1:y2, x1:x2] = 1  
    box_colored = apply_color_palette(box_mask, sym, dataset)

    return box_colored

def main(args):
    ########## nih-det
    if args.dataset == 'nih-det':
        with open(args.anno_path, 'r') as f:
            annotations = json.load(f)

        for entry in tqdm(annotations, desc='Box masking'):
            if len(entry['syms']) != 0:
                sym = entry['syms']
                img = os.path.join(args.dataset_path, entry['file_name'])

                image = Image.open(img)
                box_mask_size = (image.size[0], image.size[1])
                box_mask = np.zeros(box_mask_size, dtype=np.uint8)
                box_colored = masking(args.dataset, box_mask, entry, sym)
                box_image = Image.fromarray(box_colored.astype(np.uint8))
                blend_box_image = Image.blend(image.convert("RGBA"), box_image.convert("RGBA"), alpha=0.3)

                # save result
                img_file_name = f'{args.img_save_root}/{entry["file_name"]}'
                image.save(img_file_name)
                mask_file_name = f'{args.mask_save_root}/{entry["file_name"]}'
                blend_box_image.save(mask_file_name)

    ########## nih
    elif args.dataset == 'nih':
        annotations = pd.read_csv(args.anno_path)
        for i in tqdm(range(len(annotations)), desc='Box masking'):
            entry = annotations.loc[i]
            sym = [entry['label']]
            img = os.path.join(args.dataset_path, entry['image_name'])
            image = Image.open(img)
            box_mask_size = (image.size[0], image.size[1])
            box_mask = np.zeros(box_mask_size, dtype=np.uint8)

            box_colored = masking(args.dataset, box_mask, entry, sym)
            box_image = Image.fromarray(box_colored.astype(np.uint8))
            blend_box_image = Image.blend(image.convert("RGBA"), box_image.convert("RGBA"), alpha=0.3)

            # save result
            img_file_name = f'{args.img_save_root}/{entry["image_name"]}'
            image.save(img_file_name)
            mask_file_name = f'{args.mask_save_root}/{entry["image_name"]}'
            blend_box_image.save(mask_file_name)
            

if __name__ == '__main__':
    DATASET_PATH = 'YOUR_PATH'

    parser = argparse.ArgumentParser(description='ChestX-ray box masking (GT BBox)')
    parser.add_argument('--dataset', default='nih-det', help='nih, nih-det')
    parser.add_argument('--split', default='test', help='nih: bbox / nih-det: train,test')
    parser.add_argument('--save_root', default='./dataset/nih-det_bbox_img')
    args = parser.parse_args()

    if args.dataset == 'nih-det':
        args.dataset_path = f'{DATASET_PATH}/{args.split}' # './ChestX-det/test'
        args.anno_path = f'{PROJECT}/dataset/det_split/ChestX_Det_{args.split}.json'

    elif args.dataset == 'nih':
        args.dataset_path = f'{DATASET_PATH}/imgs' # '.nih_chest_x-rays/imgs'
        args.anno_path = f'{PROJECT}/dataset/nih_split/bbox.csv'
    
    args.img_save_root = f'{args.save_root}/{args.split}_img'
    args.mask_save_root = f'{args.save_root}/{args.split}_box'

    os.makedirs(args.img_save_root, exist_ok=True)
    os.makedirs(args.mask_save_root, exist_ok=True)
    main(args)