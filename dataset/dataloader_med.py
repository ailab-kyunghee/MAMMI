import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import json
import cv2

class Augmentation():
    def __init__(self, normalize):
        if normalize.lower() == "imagenet":
            self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif normalize.lower() == "chestx-ray":
            self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        elif normalize.lower() == "none":
            self.normalize = None
        else:
            print("mean and std for [{}] dataset do not exist!".format(normalize))
            exit(-1)

    def get_augmentation(self, augment_name, mode):
        try:
            aug = getattr(Augmentation, augment_name)
            return aug(self, mode)
        except:
            print("Augmentation [{}] does not exist!".format(augment_name))
            exit(-1)

    def basic(self, mode):
        transformList = []
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def _basic_crop(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
        else:
            transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_224(self, mode):
        transCrop = 224
        return self._basic_crop(transCrop, mode)

    def _basic_resize(self, size, mode="train"):
        transformList = []
        transformList.append(transforms.Resize(size))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_resize_224(self, mode):
        size = 224
        return self._basic_resize(size, mode)

    def _basic_crop_rot(self, transCrop, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomCrop(transCrop))
            transformList.append(transforms.RandomRotation(7))
        else:
            transformList.append(transforms.CenterCrop(transCrop))

        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
            transformList.append(self.normalize)
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def basic_crop_rot_224(self, mode):
        transCrop = 224
        return self._basic_crop_rot(transCrop, mode)

    def _full(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full(transCrop, transResize, mode)

    def full_448(self, mode):
        transCrop = 448
        transResize = 512
        return self._full(transCrop, transResize, mode)

    def _full_colorjitter(self, transCrop, transResize, mode="train"):
        transformList = []
        if mode == "train":
            transformList.append(transforms.RandomResizedCrop(transCrop))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.RandomRotation(7))
            transformList.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "val":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.CenterCrop(transCrop))
            transformList.append(transforms.ToTensor())
            if self.normalize is not None:
                transformList.append(self.normalize)
        elif mode == "test":
            transformList.append(transforms.Resize(transResize))
            transformList.append(transforms.TenCrop(transCrop))
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
            if self.normalize is not None:
                transformList.append(
                    transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
        transformSequence = transforms.Compose(transformList)

        return transformSequence

    def full_colorjitter_224(self, mode):
        transCrop = 224
        transResize = 256
        return self._full_colorjitter(transCrop, transResize, mode)


# --------------------------------------------Downstream ChestX-ray14-------------------------------------------
class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label, file
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label

class ChestX_ray14_bbox(Dataset):
    def __init__(self, data_dir, file, augment, no_normalize_aug,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.box_list = []
        self.img_label = []
        self.img_polygons = []
        self.img_name = []
        self.box_mask = []
        
        bbox_labels = pd.read_csv(file)
        for i in range(len(bbox_labels)):
            entry = bbox_labels.loc[i]
            imagePath = f'{data_dir}img/{entry["image_name"]}'
            boxPath = f'{data_dir}box/{entry["image_name"]}'
            imageLabel = entry['label']
            # imagePolygon = 
            x, y, w, h = int(entry['bbox_x']) , int(entry['bbox_y']) , int(entry['bbox_w']) , int(entry['bbox_h'])
            boxes = [x, y, x+w, y+h]
            self.img_list.append(imagePath)
            self.box_list.append(boxPath)
            self.img_label.append(imageLabel)
            self.img_polygons.append([])
            self.img_name.append(entry["image_name"])
            self.box_mask.append(boxes)

        self.augment = augment
        self.no_normalize_aug = no_normalize_aug
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        box = self.box_list[index]
        label = [self.img_label[index]]
        polygon = self.img_polygons[index]
        box_mask = self.box_mask[index]

        imageData = Image.open(file).convert('RGB')
        boxData = Image.open(box).convert('RGB')
        if self.heatmap is None:
            # imageData = self.augment(imageData)
            # boxData = self.augment(boxData)
            img = self.augment(imageData)
            box = self.augment(boxData)

            img_no_norm = self.no_normalize_aug(imageData)
            box_no_norm = self.no_normalize_aug(boxData)

            img_name = self.img_name[index]
            img_cv2 = cv2.imread(file)
            # label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, box, img_cv2, label, polygon, img_name, img_no_norm, box_no_norm, box_mask
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label
        
class ChestX_ray14_det(Dataset):
    def __init__(self, data_dir, file, augment, no_normalize_aug,
                 num_class=14, img_depth=3, heatmap_path=None,
                 pretraining=False):
        self.img_list = []
        self.box_list = []
        self.img_label = []
        self.img_polygons = []
        self.img_name = []
        self.box_mask = []
        
        with open(file, 'r') as f:
            line = json.load(f)
            for i in range(len(line)):
                lineItems = line[i]
                if len(lineItems['syms']) != 0:
                    # imagePath = os.path.join(data_dir, lineItems['file_name'])
                    imagePath = f'{data_dir}img/{lineItems["file_name"]}'
                    boxPath = f'{data_dir}box/{lineItems["file_name"]}'
                    imageLabel = lineItems['syms']
                    imagePolygon = lineItems['polygons']
                    self.img_list.append(imagePath)
                    self.box_list.append(boxPath)
                    self.img_label.append(imageLabel)
                    self.img_polygons.append(imagePolygon)
                    self.img_name.append(lineItems['file_name'])
                    self.box_mask.append(lineItems['boxes'])

        # with open(file, "r") as fileDescriptor:
        #     line = True
        #     while line:
        #         line = fileDescriptor.readline()
        #         if line:
        #             lineItems = line.split()
        #             imagePath = os.path.join(data_dir, lineItems[0])
        #             imageLabel = lineItems[1:num_class + 1]
        #             imageLabel = [int(i) for i in imageLabel]
        #             self.img_list.append(imagePath)
        #             self.img_label.append(imageLabel)

        self.augment = augment
        self.no_normalize_aug = no_normalize_aug
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        file = self.img_list[index]
        box = self.box_list[index]
        label = self.img_label[index]
        polygon = self.img_polygons[index]
        box_mask = self.box_mask[index]

        imageData = Image.open(file).convert('RGB')
        boxData = Image.open(box).convert('RGB')
        if self.heatmap is None:
            # imageData = self.augment(imageData)
            # boxData = self.augment(boxData)
            img = self.augment(imageData)
            box = self.augment(boxData)

            img_no_norm = self.no_normalize_aug(imageData)
            box_no_norm = self.no_normalize_aug(boxData)

            img_name = self.img_name[index]
            img_cv2 = cv2.imread(file)
            # label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, box, img_cv2, label, polygon, img_name, img_no_norm, box_no_norm, box_mask
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform):
        self.data_dir = data_dir
        self.phase = phase

        self.classes = ['normal', 'positive', 'pneumonia', 'COVID-19']
        self.class2label = {c: i for i, c in enumerate(self.classes)}

        # collect training/testing files
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                self.datalist.append((os.path.join(data_dir, 'train', fname), label))
            else:
                self.datalist.append((os.path.join(data_dir, 'test', fname), label))

        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        label = self.class2label[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


# class Node21(torch.utils.data.Dataset):
#     def __init__(self, data_dir, phase, transform):
#         self.data_dir = data_dir
#         self.phase = phase

#         if phase == 'train':
#             with open(os.path.join(data_dir, 'train_mae.txt')) as f:
#                 fnames = f.readlines()
#         elif phase == 'test':
#             with open(os.path.join(data_dir, 'test_mae.txt')) as f:
#                 fnames = f.readlines()
#         fnames = [fname.strip() for fname in fnames]

#         self.datalist = list()
#         for line in fnames:
#             fname, label = line.split(' ')
#             self.datalist.append((os.path.join(data_dir, 'images', fname), int(label)))
#         # metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
#         # self.datalist = list()
#         # for i in range(len(metadata)):
#         #     fname = metadata.loc[i, 'img_name']
#         #     if fname in fnames:
#         #         label = metadata.loc[i, 'label']
#         #         self.datalist.append((os.path.join(data_dir, 'images', fname), label))

#         # transforms
#         self.transform = transform

#     def __len__(self):
#         return len(self.datalist)

#     def __getitem__(self, index):
#         fpath, label = self.datalist[index]
#         image, _ = medpy.io.load(fpath)
#         image = image.astype(np.float)
#         image = (image - image.min()) / (image.max() - image.min())
#         image = (image * 255).astype(np.uint8)
#         image = image.transpose(1, 0)
#         image = Image.fromarray(image).convert('RGB')
#         image = self.transform(image)
#         label = torch.tensor([label], dtype=torch.float32)
#         return image, label


class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False
                 ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')

        else:
            self.heatmap = None

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        # image = cv2.imread(self._images_list[idx], 0)
        # image = Image.fromarray(image)
        # if self.mode == 'train':
        #     image = self.transform(image)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #
        # # resize and normalize; e.g., ToTensor()
        # image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # image = image / 255.0
        # __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        # __std__ = np.array([[[0.229, 0.224, 0.225]]])
        # image = (image - __mean__) / __std__

        if self.heatmap is None:
            image = Image.open(self._images_list[idx]).convert('RGB')

            image = self.transform(image)

            # image = image.transpose((2, 0, 1)).astype(np.float32)

            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return image, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            image = Image.open(self._images_list[idx]).convert('RGB')
            image, heatmap = self.transform(image, heatmap)
            heatmap = heatmap.permute(1, 2, 0)
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return [image, heatmap], label


'''
 NIH:(train:75312  test:25596)
 0:A 1:Cd 2:Ef 3:In 4:M 5:N 6:Pn 7:pnx 8:Co 9:Ed 10:Em 11:Fi 12:PT 13:H
 Chexpert:(train:223415 val:235)
 0:NF 1:EC 2:Cd 3:AO 4:LL 5:Ed 6:Co 7:Pn 8:A 9:Pnx 10:Ef 11:PO 12:Fr 13:SD
 combined:
 0: Airspace Opacity(AO)	1: Atelectasis(A)	2:Cardiomegaly(Cd)	3:Consolidation(Co)
 4:Edema(Ed)	5:Effusion(Ef)	6:Emphysema(Em)	7:Enlarged Card(EC)	8:Fibrosis(Fi)	
 9:Fracture(Fr)	10:Hernia(H)	11:Infiltration(In)	12:Lung lession(LL)	13:Mas(M)	
 14:Nodule(N)	15:No finding(NF)	16:Pleural thickening(PT)	17:Pleural other(PO)	18:Pneumonia(Pn)	
 19:Pneumothorax(Pnx)	20:Support Devices(SD)
'''
