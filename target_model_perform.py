import argparse
import torch
from tqdm import tqdm
from torchvision import models 
from dataset.dataloader_med import Augmentation, ChestX_ray14
import numpy as np 
import pandas as pd
import os
import sklearn.metrics as sklm
from sklearn.metrics._ranking import roc_auc_score

PROJECT = os.path.dirname(os.path.realpath(__file__))

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    # print(dataGT.shape, dataPRED.shape)
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    print(outAUROC)
    return outAUROC

@torch.no_grad()
def evaluate_chestxray(args, data_loader, model, device):
    model.eval()
    outputs = []
    targets = []
    for batch in tqdm(data_loader,desc='Inference'):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

        outputs.append(output)
        targets.append(target)

    num_classes = 14 #args.nb_classes

    outputs = torch.cat(outputs, dim=0).sigmoid().cpu().numpy()

    targets = torch.cat(targets, dim=0).cpu().numpy()
    print(targets.shape, outputs.shape)
    
    np.save(f'{args.save_dir}/npy_file/y_gt.npy', targets)
    np.save(f'{args.save_dir}/npy_file/y_pred.npy', outputs)

    auc_each_class = computeAUROC(targets, outputs, num_classes)
    auc_each_class_array = np.array(auc_each_class)
    missing_classes_index = np.where(auc_each_class_array == 0)[0]

    if missing_classes_index.shape[0] > 0:
        print('There are classes that not be predicted during testing,'
              ' the indexes are:', missing_classes_index)

    auc_avg = np.average(auc_each_class_array[auc_each_class_array != 0])

    return {'auc_avg': auc_avg, 'auc_each_class': auc_each_class}

def make_pred_multilabel(args, val_loader, test_loader, model, device):
    with open(args.gt_label, 'r') as f: 
        PRED_LABEL = (f.read()).split('\n')
    
    pred_df = pd.DataFrame()
    bi_pred_df = pd.DataFrame()
    true_df = pd.DataFrame()

    for mode in ['Threshold', 'test']:
    # for mode in ['test']:
        if mode == "Threshold":
            loader = val_loader
            Eval_df = pd.DataFrame(columns=["label", 'bestthr'])
            thrs = []

        if mode == "test":
            loader = test_loader
            TestEval_df = pd.DataFrame(columns=["label", 'auc', "auprc"])

            Eval = pd.read_csv(f'{args.save_dir}/results/Threshold.csv')
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

        for batch in tqdm(loader,desc='Predict for compute threshold'):
            inputs = batch[0]
            labels = batch[1]
            

            inputs = inputs.to(device)
            labels = labels.to(device)

            true_labels = labels.cpu().data.numpy()

            batch_size = true_labels.shape

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.sigmoid()
                probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size[0]):
                thisrow = {}
                bi_thisrow = {}
                truerow = {}

                for k in range(len(PRED_LABEL)):
                    thisrow["prob_" + PRED_LABEL[k]] = probs[j, k]
                    truerow[PRED_LABEL[k]] = true_labels[j, k]

                    if mode == "test":
                       bi_thisrow["bi_" + PRED_LABEL[k]] = probs[j, k] >= thrs[k]

                pred_df = pd.concat([pred_df, pd.DataFrame([thisrow])], ignore_index=True)
                true_df = pd.concat([true_df, pd.DataFrame([truerow])], ignore_index=True)
                if mode == "test":
                    bi_pred_df = pd.concat([bi_pred_df, pd.DataFrame([bi_thisrow])], ignore_index=True)


        for column in true_df:
            if column not in PRED_LABEL:
                continue
            actual = true_df[column]
            pred = pred_df["prob_" + column]
            
            thisrow = {}
            thisrow['label'] = column
            
            if mode == "test":
                bi_pred = bi_pred_df["bi_" + column]            
                thisrow['auc'] = np.nan
                thisrow['auprc'] = np.nan
            else:
                thisrow['bestthr'] = np.nan

            try:
                if mode == "test":
                    thisrow['auc'] = sklm.roc_auc_score( 
                        actual.to_numpy().astype(int), pred.to_numpy())

                    thisrow['auprc'] = sklm.average_precision_score(
                        actual.to_numpy().astype(int), pred.to_numpy())
                else:
                    # p: precision, r: recall, t: thresholds
                    # p, r, t = sklm.precision_recall_curve(actual.as_matrix().astype(int), pred.as_matrix())
                    p, r, t = sklm.precision_recall_curve(actual.to_numpy().astype(int), pred.to_numpy())
                    # Choose the best threshold based on the highest F1 measure
                    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
                    bestthr = t[np.where(f1 == max(f1))]

                    thrs.append(bestthr)
                    thisrow['bestthr'] = bestthr[0]


            except BaseException:
                print("can't calculate auc for " + str(column))

            if mode == "Threshold":
                Eval_df = pd.concat([Eval_df, pd.DataFrame([thisrow])], ignore_index=True)

            if mode == "test":
                # TestEval_df = TestEval_df.append(thisrow, ignore_index=True)
                TestEval_df = pd.concat([TestEval_df, pd.DataFrame([thisrow])], ignore_index=True)

        pred_df.to_csv(f'{args.save_dir}/results/preds.csv', index=False)
        true_df.to_csv(f'{args.save_dir}/results/True.csv', index=False)


        if mode == "Threshold":
            Eval_df.to_csv(f'{args.save_dir}/results/Threshold.csv', index=False)

        if mode == "test":
            TestEval_df.to_csv(f'{args.save_dir}/results/TestEval.csv', index=False)
            bi_pred_df.to_csv(f'{args.save_dir}/results/bipred.csv', index=False)
    
    print("AUC avg:", TestEval_df['auc'].sum() / 14.0)

    print("done")

    return pred_df, Eval_df, bi_pred_df, TestEval_df  # , bi_pred_df , Eval_bi_df

def main(args):
    ####### checkpoint & model load
    if args.model == 'densenet121':
        checkpoint = torch.load(f'{PROJECT}/pretrained/target_model/densenet121_CXR_0.3M_mocov2.pth', map_location='cpu')
        # model = models.__dict__[checkpoint['args'].model](num_classes=checkpoint['args'].nb_classes)
        model = models.__dict__['densenet121'](num_classes=14)

    # print("Load pre-trained checkpoint from: %s" % args.finetune)
    if 'state_dict' in checkpoint.keys():
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f'Model weigth load : {msg}')
    model.to('cuda')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    # model.classifier._parameters['weight']
    print(f'Model weigth load : {msg}')

    ####### Dataset
    transform = Augmentation(normalize="chestx-ray").get_augmentation("full_224", "val")
    print(f'Dataset: {args.dataset}')
    if args.dataset == 'nih-14':
        valid_data = f'{PROJECT}/dataset/nih_split/val_official.txt'
        test_data = f'{PROJECT}/dataset/nih_split/test_official.txt'

        val_dataset = ChestX_ray14(args.data_path, valid_data, augment=transform, num_class=14)
        test_dataset = ChestX_ray14(args.data_path, test_data, augment=transform, num_class=14)
    else:
        raise NotImplementedError

    sampler_valid = torch.utils.data.SequentialSampler(val_dataset)
    data_loader_valid = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_valid,
        batch_size=128, #args.batch_size,
        num_workers=4, #args.num_workers,
        pin_memory=True, #args.pin_mem,
        drop_last=False
    )

    sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=128, #args.batch_size,
        num_workers=4, #args.num_workers,
        pin_memory=True, #args.pin_mem,
        drop_last=False
    )

    #### evaluation
    exist_npy = True
    args.save_dir = f'{PROJECT}/{args.save_dir}/{args.model}'
    pred_file = f'{args.save_dir}/npy_file/y_pred.npy'
    if not os.path.exists(pred_file):
        os.makedirs(args.save_dir, exist_ok=True)
        exist_npy = False
        print(f'npy exist: {exist_npy}')
    else:
        os.makedirs(f'{args.save_dir}/results', exist_ok=True)
        exist_npy = True
        print(f'npy exist: {exist_npy}')
    
    if exist_npy == False:
        test_stats = evaluate_chestxray(args, data_loader_test, model, device='cuda')
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        exist_npy = True
    
    if exist_npy == True:
        if args.dataset == 'nih-14':
            args.gt_label = f'{PROJECT}/dataset/nih_split/nih_labels.txt'
        make_pred_multilabel(args, data_loader_valid, data_loader_test, model, device='cuda')

if __name__ == '__main__':
    NIH14_DATA_PATH = 'YOUR_PATH/nih_chest_x-rays/imgs'
    parser = argparse.ArgumentParser(description='Model verification')
    parser.add_argument('--dataset', default='nih-14')
    parser.add_argument('--model', default='densenet121', help='densenet121, vit-b16')
    parser.add_argument('--save_dir', default='results/model_perform')
    args = parser.parse_args()

    args.split = 'test'
    if args.dataset == 'nih-14':
        args.data_path = NIH14_DATA_PATH
    
    main(args)