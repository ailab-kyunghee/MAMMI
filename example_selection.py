from torchvision import models
from models import densenet
from torchvision.utils import save_image
import torch
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import os
import json 
from PIL import Image
import pandas as pd
from dataset.dataloader_med import Augmentation, ChestX_ray14
from collections import Counter

NIH_CLASS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
             'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
             'Emphysema', 'Fibrosis', 'Pleural thickening', 'Hernia']

NIH14_DATA_PATH = 'YOUR_PATH/nih_chest_x-rays/imgs'
def setting(args):
    # dataset path / target label path / csv path
    if args.dataset == 'nih-14':
        args.data_root_img = NIH14_DATA_PATH
        if not os.path.exists(args.data_root_img):
            print('Please check the dataset path')

        if args.split == 'train':
            args.split_path = './dataset/nih_split/train_official.txt'
        elif args.split == 'val':
            args.split_path = './dataset/nih_split/val_official.txt'
        elif args.split == 'test':
            args.split_path = './dataset/nih_split/test_official.txt'
    
    args.pkl_save_root = os.path.join(args.save_root, args.model, 'activation')
    args.base_root = os.path.join(args.save_root, args.model, f'{args.dataset}_adap_{args.adaptive_percent}_examples', args.layer)
    args.img_save_root = os.path.join(args.base_root, 'images')

def load_model(target_model, dataset, layer):
    if dataset == 'nih-14':
        if target_model == 'densenet121':
            checkpoint = torch.load('./pretrained/target_model/densenet121_CXR_0.3M_mocov2.pth', map_location='cpu')
            # model = models.__dict__['densenet121'](num_classes=14)
            model = densenet.__dict__['densenet121'](num_classes=14)
            if layer == 'penultimate':
                model.classifier = torch.nn.Identity()

        elif target_model == 'resnet50':
            checkpoint = torch.load('./pretrained/target_model/resnet50_imagenet_mocov2.pth', map_location='cpu')
            model = models.__dict__['resnet50'](num_classes=14)
            if layer == 'layer4':
                model.fc = torch.nn.Identity()
            elif layer == 'layer3':
                model.layer4 = torch.nn.Identity()
                model.fc = torch.nn.Identity()
            elif layer == 'layer2':
                model.layer3 = torch.nn.Identity()
                model.layer4 = torch.nn.Identity()
                model.fc = torch.nn.Identity()
            elif layer == 'layer1':
                model.layer2 = torch.nn.Identity()
                model.layer3 = torch.nn.Identity()
                model.layer4 = torch.nn.Identity()
                model.fc = torch.nn.Identity()

        if 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('Load state dict', msg)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    model.to('cuda') # or model.cuda()
    model.eval()

    return model

def compute_activation(args, model, save_dir):
    if os.path.exists(save_dir):
        with open(save_dir, 'rb') as f:
            act_matrix = pickle.load(f)
        return act_matrix
    
    elif not os.path.exists(args.pkl_save_root):
        os.makedirs(args.pkl_save_root, exist_ok=True)

    print(f'Phase1. Save activation: {save_dir}')
    if args.dataset == 'nih-14':
        transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")
        traindata = ChestX_ray14(args.data_root_img, args.split_path, augment=transform, num_class=14)

        sampler_test = torch.utils.data.SequentialSampler(traindata)
        trainloader = torch.utils.data.DataLoader(
            traindata, sampler=sampler_test,
            batch_size=128, #args.batch_size,
            num_workers=4, #args.num_workers,
            pin_memory=True, #args.pin_mem,
            drop_last=False,
            shuffle=False
        )

    with torch.no_grad():
        act_matrix = []
        counter = 0
        example_name = []
        for batch_idx, samples in enumerate(tqdm(trainloader)):    
            # samples['lab'] : pathologies labels
            # samples['img'] : image
            # samples['idx'] : index
            if args.dataset == 'chexpert' or args.dataset == 'chexpert-small':
                image = samples['img'].cuda()
            elif args.dataset == 'nih-14':
                image = samples[0].cuda()
                example_name = example_name + samples[2]
            act_matrix.append(model(image).squeeze().cpu().detach().numpy())
            
            ### if use actvation slicing
            if batch_idx % int(len(trainloader)/args.num_slicing) == 0 and batch_idx != 0:
                act_matrix = np.concatenate(act_matrix, axis=0)
                with open(f"{save_dir}/slice_act_{counter}_{args.layer}", 'wb') as f:
                    pickle.dump(act_matrix, f)
                with open(f"{save_dir}/slice_act_{counter}_{args.layer}.json",'w') as f:
                    json.dump(example_name, f, indent=4)               
                counter += 1
                act_matrix = []
        
        if args.num_slicing == 1:
            act_matrix = np.concatenate(act_matrix, axis=0)
            with open(save_dir, 'wb') as f:
                pickle.dump(act_matrix, f)
            with open(save_dir.replace('pkl', 'json'),'w') as f: # only crop 
                json.dump(example_name, f, indent=4)
    
    print('End of Phase1. Save activation')
    return act_matrix

def make_idx_matrix(args, size, save_dir, idx_feat_dir):
    if not os.path.exists(save_dir):
    # if True:
        print(f'Phase3. Save sorted index file : {save_dir}')
        os.makedirs(save_dir, exist_ok=True)

        idx_mats = []
        feat_mats = []
        idx_matrix = np.zeros((args.num_example, size))

        for i in range(args.num_slicing):
            with open(f"{idx_feat_dir}/idx_{args.num_example}_{i}_{args.layer}.pkl", 'rb') as f:
                idx_matrix = pickle.load(f)
            with open(f"{idx_feat_dir}/feat_{args.num_example}_{i}_{args.layer}.pkl", 'rb') as f:
                feat_matrix = pickle.load(f)
            idx_mats.append(idx_matrix)
            feat_mats.append(feat_matrix)

        idx_mats = np.concatenate(idx_mats, axis=0)
        feat_mats = np.concatenate(feat_mats, axis=0)

        for j in range(idx_matrix.shape[1]):
            sorted_idx = np.argsort(feat_mats[:,j])
            top_idx = np.flip(sorted_idx[-args.num_example:])
            idx_matrix[:,j] = idx_mats[top_idx,j]
        
        with open(f"{save_dir}/slice_idx_{args.num_example}_{args.layer}.pkl", 'wb') as f:
            pickle.dump(idx_matrix, f)

        print('End of Phase3. Save sorted index file')
    else:
        print('Phase3 sorted index file exists')
        
        with open(f"{save_dir}/slice_idx_{args.num_example}_{args.layer}.pkl", 'rb') as f:
            idx_matrix = pickle.load(f)
    
    return idx_matrix

def main(args):    
    setting(args)
    ## Load model ##
    model = load_model(args.model, args.dataset, args.layer)
    
    activation_path = os.path.join(args.pkl_save_root, f'{args.layer}_act.pkl')
    # if True:
    if not os.path.exists(activation_path):
        act_matrix = compute_activation(args, model, activation_path)
    else:
        act_matrix = compute_activation(args, model, activation_path)
        print(f'Phase1 activation file exists, shape = {act_matrix.shape}')
        
    print(f'Phase2. Save example images: {args.img_save_root}')
    if not os.path.exists(args.img_save_root):
        os.makedirs(args.img_save_root)
    
    if args.dataset == 'nih-14':
        pathol_class = NIH_CLASS
        no_transform = Augmentation(normalize="none").get_augmentation("full_224", "val")
        traindata = ChestX_ray14(args.data_root_img, args.split_path, augment=no_transform, num_class=14)# no normalize
        transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")
        trainset = ChestX_ray14(args.data_root_img, args.split_path, augment=transform, num_class=14)
        
    img_label_dict = {}
    column = []
    
    num_img = []
    inter = []
    thres = []

    max_act = []
    min_act = []
    count_label_list = []

    for i in tqdm(range(act_matrix.shape[1]), desc='Save example images'):

        os.makedirs(f'{args.img_save_root}/{i:04d}', exist_ok=True)
        ### top activation - sorting
        target_act = act_matrix[:,i] # i-th class activation
        max_act.append(max(target_act))
        min_act.append(min(target_act))

        interval = max(target_act) - min(target_act)
        threshold = max(target_act) - interval * (1 - args.adaptive_percent / 100)

        sorted_act = np.sort(target_act)[::-1]
        sort_act = sorted_act[sorted_act > threshold]
        num_img.append(len(sort_act))

        sorted_idx = np.argsort(target_act)[::-1]
        sorted_idx = sorted_idx[:len(sort_act)]

        for j in range(len(sorted_idx)):
            img_dir = f'{args.img_save_root}/{i:04d}/{i:04d}_{j:02d}.jpg'
            if args.dataset == 'nih-14':
                labels = traindata[int(sorted_idx[j])][1].numpy()
                for target_idx, l in enumerate(labels):
                    if l == 1:
                        count_label_list.append(pathol_class[target_idx])
                image = traindata[int(sorted_idx[j])][0]
                ind = int(sorted_idx[j])
                sa = trainset[int(sorted_idx[j])][0].cuda()

            out = model(sa.unsqueeze(0)).cpu().detach().numpy()
            label_out_dict = dict(zip(pathol_class, zip(out[0], labels)))
            finding = []
            patholo = list(label_out_dict.keys())
            for n in range(len(list(label_out_dict.values()))):
                v = list(label_out_dict.values())[n]
                if v[1] == 1.0:
                    finding.append(patholo[n])
            if len(finding) == 0:
                finding.append('No finding')
            label_out_dict['Findings'] = finding
            if args.layer == 'fc':
                label_out_dict['Class'] = pathol_class[i]
            label_out_dict['Img idx'] = ind

            if column != list(label_out_dict.keys()) and i != 0:
                print('error')

            if args.dataset == 'chexpert' or args.dataset == 'chexpert-small':
                Image.fromarray(image).save(img_dir)
            elif args.dataset == 'nih-14':
                save_image(image, img_dir)
            # img_label_dict[f'{i:04d}/{i:04d}_{j:02d}.jpg'] = list(labels)
            img_label_dict[f'{i:04d}/{i:04d}_{j:02d}.jpg'] = list(label_out_dict.values())
            column = list(label_out_dict.keys())
    count_label = Counter(count_label_list)
    avg_num_img = sum(num_img) / len(num_img)
    print(f'Number of each neuron images: {num_img}')
    print(f'Average number of each neuron images: {avg_num_img}')

    num_example_path = os.path.join(args.base_root, 'num_example.pkl')
    with open(num_example_path, 'wb') as f:
        pickle.dump(num_img, f)

    df = pd.DataFrame.from_dict(data=img_label_dict, orient='index', columns=column)
    if args.dataset == 'chexpert' or args.dataset == 'chexpert-small':
        df = df[['Class', 'Img idx', 'Findings', 'Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', 'Effusion', 'Pneumonia', 'Cardiomegaly']]
    elif args.dataset == 'nih-14':
        if args.layer == 'fc':
            df = df[['Class', 'Img idx', 'Findings', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                        'Edema', 'Emphysema', 'Fibrosis', 'Pleural thickening', 'Hernia']]
        elif args.layer == 'penultimate':
            if args.component_example == 'crop':
                df = df[['Img idx', 'Findings']]
            elif args.component_example == 'orig':
                df = df[['Img idx', 'Findings', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                        'Edema', 'Emphysema', 'Fibrosis', 'Pleural thickening', 'Hernia']]
    csv_path = os.path.join(args.base_root, 'img_label_pair.csv')
    df.to_csv(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example selection')
    parser.add_argument('--dataset', default='nih-14')
    parser.add_argument('--split', default='train', help='train / val for image dataset')
    
    parser.add_argument('--model', default='densenet121', help='densenet121 / resnet50 / vit-b16')
    parser.add_argument('--layer', default='fc', help='fc / penultimate')
    # parser.add_argument('--model', default='resnet50', help='densenet121 / resnet50 / vit-b')
    # parser.add_argument('--layer', default='fc', help='fc / layer4 / layer3 / layer2 / layer1')
    parser.add_argument('--num_slicing', default=1, type=int, help='Activation slicing for save')
    parser.add_argument('--save_root', default='./results/examples', help='Path to saved result')
    parser.add_argument('--adaptive_percent', default='93', type=int, help='85~95')

    args = parser.parse_args()

    main(args)