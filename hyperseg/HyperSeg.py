import os
from copy import deepcopy
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from .MambaHSI import MambaHSI
from sklearn.mixture import GaussianMixture
from .Utils import build_segov, perform_segearth, RandomTransformWithRestoreTensor, \
    output_metric, gmm_cluster_1d, multivariate_gaussian_pdf, \
    reduce_image_channels, rsshow
from scipy.ndimage import zoom
from sklearn.metrics import *


def segov_classification(image, tar_image, label_list, args):
    segov = build_segov(label_list)

    logits_seg = np.zeros((image.shape[0], image.shape[1], len(np.unique(tar_image)) - 1))
    if args.data_name == 'PaviaC':
        scale_list = [1, 2]
        slide_crop_list = [224, 224]
        slide_stride_list = [112, 112]
    elif args.data_name == 'chikusei':
        scale_list = [1, 2]
        slide_crop_list = [224, 224]
        slide_stride_list = [112, 112]
    elif args.data_name == 'AeroRIT':
        scale_list = [1, 2]
        slide_crop_list = [448, 448]
        slide_stride_list = [224, 224]
    else:
        raise ValueError

    for scale, slide_crop, slide_stride in zip(scale_list, slide_crop_list, slide_stride_list):
        image_resized = zoom(image, (scale, scale, 1), order=3)
        segov.slide_crop = slide_crop
        segov.slide_stride = slide_stride
        pre_seg_, logits_seg_ = perform_segearth(segov, image_resized)
        logits_seg_ = zoom(logits_seg_.astype(np.float32), (1/scale, 1/scale, 1), order=3)
        logits_seg += logits_seg_ / len(scale_list)
    pre_seg = logits_seg.argmax(-1) + 1

    return pre_seg, logits_seg

def build_models(band, class_num, args):
    model_main = MambaHSI(in_channels=band, num_classes=class_num, hidden_dim=16)
    model_main.cuda()

    # optimizer
    optimizer_main = optim.Adam(model_main.parameters(), lr=args.learning_rate,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # learning scheduler
    total_iters = args.epoches*args.iterations
    scheduler_main = optim.lr_scheduler.CosineAnnealingLR(optimizer_main, T_max=total_iters, eta_min=args.eta_min)

    return model_main, optimizer_main, scheduler_main

def sampling_random_samples(pesudo_label, pesudo_logits, sample_weight, sample_num):
    y_random_samples = np.zeros_like(pesudo_logits)
    lw_random_samples = np.zeros_like(pesudo_label).astype(np.float32)

    random_samples_positions = []
    p = 1
    for t in np.unique(pesudo_label[pesudo_label > 0]):
        positions = np.vstack(np.where(pesudo_label == t)).T
        sample_weight_ = sample_weight[positions[:, 0], positions[:, 1]].astype(np.float64) ** p
        if sample_weight_.max() == 0:
            sample_weight_ = np.ones(len(sample_weight_)) / len(sample_weight_)
        else:
            sample_weight_ = sample_weight_ / sample_weight_.sum()
        ind = np.random.choice(range(len(positions)), sample_num, p=sample_weight_)
        sample_positions = positions[ind]
        random_samples_positions.append(sample_positions)
    random_samples_positions = np.concatenate(random_samples_positions, axis=0)
    y_random_samples[random_samples_positions[:, 0], random_samples_positions[:, 1]] = \
        pesudo_logits[random_samples_positions[:, 0], random_samples_positions[:, 1]]
    lw_random_samples[random_samples_positions[:, 0], random_samples_positions[:, 1]] = 1

    y_random_samples = torch.from_numpy(y_random_samples)
    lw_random_samples = torch.from_numpy(lw_random_samples)
    return y_random_samples, lw_random_samples

def sampling_clean_samples(pre_model_main, BvSB_model_main,
                    image_hsi, tar_image, class_num, label_list, image_hsi_pca):
    positions_positive_dict, positions_negative_dict = {}, {}
    for t in np.unique(pre_model_main[pre_model_main > 0]):
        positions = np.stack(np.where((pre_model_main == t) & (tar_image > 0))).T

        bvsb = BvSB_model_main[positions[:, 0], positions[:, 1]]
        # if len(bvsb) < 5:
        #     continue

        separate_labels, gmm_mean, gmm_std = gmm_cluster_1d(bvsb, n_components=3, plot_flag=False)
        ind_high_confidence = np.where(separate_labels == gmm_mean.argmax())[0]
        ind_low_confidence = np.where(separate_labels == gmm_mean.argmin())[0]

        ind_high_confidence = np.random.choice(ind_high_confidence, size=len(ind_high_confidence)//5, replace=False)
        # if len(ind_high_confidence) < 5:
        #     continue
        positions_positive_dict[t] = positions[ind_high_confidence]
        positions_negative_dict[t] = positions[ind_low_confidence]

    gmm_positives = {}
    for k, v in positions_positive_dict.items():
        gmm = GaussianMixture(n_components=5)
        # gmm = BayesianGaussianMixture(n_components=10, covariance_type='full', max_iter=1000)
        gmm.fit(image_hsi_pca[v[:, 0], v[:, 1]])
        gmm_positives[k] = deepcopy(gmm)

    T = 1
    probs_dict_positive = {}
    for k, v in positions_positive_dict.items():
        spectra_pca = image_hsi_pca[v[:, 0], v[:, 1]]
        pdfs = np.zeros((len(spectra_pca), class_num))
        for kk, vv in gmm_positives.items():
            pdfs[:, kk - 1] = np.exp(vv.score_samples(spectra_pca) / T)
        probs_dict_positive[k] = pdfs / pdfs.sum(-1, keepdims=True)

    
    probs_dict_negative = {}
    for k, v in positions_negative_dict.items():
        spectra_pca = image_hsi_pca[v[:, 0], v[:, 1]]
        pdfs = np.zeros((len(spectra_pca), class_num))
        for kk, vv in gmm_positives.items():
            pdfs[:, kk - 1] = np.exp(vv.score_samples(spectra_pca) / T)
        probs_dict_negative[k] = pdfs / pdfs.sum(-1, keepdims=True)

    y_clean_samples = np.zeros((tar_image.shape[0], tar_image.shape[1], class_num)).astype(np.float32)
    lw_clean_samples = np.zeros_like(tar_image).astype(np.float32)
    for k, v in probs_dict_positive.items():
        positions = positions_positive_dict[k]
        y_clean_samples[positions[:, 0], positions[:, 1]] = v
        lw_clean_samples[positions[:, 0], positions[:, 1]] = 1
    y_clean_samples = torch.from_numpy(y_clean_samples)
    lw_clean_samples = torch.from_numpy(lw_clean_samples)

    y_rectified_samples = np.zeros((tar_image.shape[0], tar_image.shape[1], class_num)).astype(np.float32)
    lw_rectified_samples = np.zeros_like(tar_image).astype(np.float32)
    for k, v in probs_dict_negative.items():
        positions = positions_negative_dict[k]
        y_rectified_samples[positions[:, 0], positions[:, 1]] = v
        lw_rectified_samples[positions[:, 0], positions[:, 1]] = 1
    y_rectified_samples = torch.from_numpy(y_rectified_samples)
    lw_rectified_samples = torch.from_numpy(lw_rectified_samples)

    return y_clean_samples, lw_clean_samples, y_rectified_samples, lw_rectified_samples


def hyperseg_classification(image, image_hsi, label_list, tar_image, patch_positions, args):
    pre_seg, logits_seg = segov_classification(image, tar_image, label_list, args)
    logits_sorted = np.sort(logits_seg, axis=-1)
    BvSB_seg = logits_sorted[..., -1] - logits_sorted[..., -2]

    # data prepare and models
    image_hsi_pca = reduce_image_channels(image_hsi, min(5, image_hsi.shape[-1]))
    x_main = torch.from_numpy(image_hsi).permute(2, 0, 1).unsqueeze(0).float().cuda()
    class_num, sample_num, warmup_epochs = len(label_list), args.samples, args.warmup_epoches
    pre_model_main, BvSB_model_main = np.copy(pre_seg), np.ones_like(BvSB_seg)


    model_main, optimizer_main, scheduler_main = \
        build_models(image_hsi.shape[-1], class_num, args)
    criterion_ce = nn.CrossEntropyLoss(reduction='none')


    # augmentation
    augmentor_weak = RandomTransformWithRestoreTensor(add_noise_ratio=0.1, mask_pixel_ratio=0.0)

    OA_main_epoches, AA_main_epoches, Kappa_main_epoches = [], [], []
    Recall_main_epoches, Precision_main_epoches = [], []
    pbar = tqdm(range(args.epoches))
    for epoch in pbar:
        model_main.train()

        pesudo_label = pre_seg
        pesudo_logits = (logits_seg == logits_seg.max(-1, keepdims=True)).astype(np.float32)
        sample_weight = np.copy(BvSB_seg).astype(np.float64)

        if epoch < warmup_epochs:
            y_rectified_samples = np.zeros_like(pesudo_logits)
            lw_rectified_samples = np.zeros_like(tar_image)
            y_rectified_samples = torch.from_numpy(y_rectified_samples)
            lw_rectified_samples = torch.from_numpy(lw_rectified_samples)

            y_clean_samples = np.zeros_like(pesudo_logits)
            lw_clean_samples = np.zeros_like(tar_image)
            y_clean_samples = torch.from_numpy(y_clean_samples)
            lw_clean_samples = torch.from_numpy(lw_clean_samples)
        else:
            y_clean_samples, lw_clean_samples, y_rectified_samples, lw_rectified_samples = \
                sampling_clean_samples(pre_model_main, BvSB_model_main,
                                   image_hsi, tar_image, class_num,
                                   label_list, image_hsi_pca)


        for iter in range(args.iterations):
            y_random_samples_weak, lw_random_samples_weak = \
                sampling_random_samples(pesudo_label, pesudo_logits, sample_weight, sample_num)

            patch_positions_iter = np.copy(patch_positions)
            for idx, (upper, bottom, left, right) in enumerate(patch_positions_iter):
                y_random_samples_ = y_random_samples_weak[upper:bottom, left:right].cuda()
                lw_random_samples_ = lw_random_samples_weak[upper:bottom, left:right].cuda()

                y_rectified_samples_ = y_rectified_samples[upper:bottom, left:right].cuda()
                lw_rectified_samples_ = lw_rectified_samples[upper:bottom, left:right].cuda()

                y_clean_samples_ = y_clean_samples[upper:bottom, left:right].cuda()
                lw_clean_samples_ = lw_clean_samples[upper:bottom, left:right].cuda()

                mask_random = lw_random_samples_ > 0
                mask_rectified = lw_rectified_samples_ > 0
                mask_clean = lw_clean_samples_ > 0
                
                x_main_ = x_main[..., upper:bottom, left:right].cuda()
                x_main_aug = augmentor_weak(x_main_)
                logits_model_main_aug, reconstruction_aug = model_main(x_main_aug)
                logits_model_main = augmentor_weak.restore(logits_model_main_aug)

                if mask_random.max() > 0:
                    logits_model_main_ = logits_model_main[..., mask_random][0].T
                    loss_random = criterion_ce(logits_model_main_, y_random_samples_[mask_random])
                    loss_random = (loss_random * lw_random_samples_[mask_random]).mean()
                else:
                    loss_random = 0
                
                if mask_rectified.max() > 0:
                    logits_model_main_ = logits_model_main[..., mask_rectified][0].T
                    loss_rectified = criterion_ce(logits_model_main_, y_rectified_samples_[mask_rectified])
                    loss_rectified = (loss_rectified * lw_rectified_samples_[mask_rectified]).mean()
                else:
                    loss_rectified = 0

                if mask_clean.max() > 0:
                    logits_model_main_ = logits_model_main[..., mask_clean][0].T
                    loss_clean = criterion_ce(logits_model_main_, y_clean_samples_[mask_clean])
                    loss_clean = (loss_clean * lw_clean_samples_[mask_clean]).mean()
                else:
                    loss_clean = 0

                loss = loss_random + args.lambda1 * loss_rectified + args.lambda2 * loss_clean
                if loss > 0:
                    optimizer_main.zero_grad()
                    loss.backward()
                    optimizer_main.step()
                    scheduler_main.step()
                else:
                    continue

        torch.cuda.empty_cache()
        model_main.eval()
        logits_model_main = np.zeros_like(logits_seg)
        for idx, (upper, bottom, left, right) in enumerate(patch_positions):
            with torch.no_grad():
                logits_model_main_ = model_main(x_main[..., upper:bottom, left:right])
                logits_model_main_ = nn.functional.softmax(logits_model_main_, dim=1)
                logits_model_main_ = logits_model_main_[0].permute(1, 2, 0).cpu().numpy()
            logits_model_main[upper:bottom, left:right] = logits_model_main_
        pre_model_main = logits_model_main.argmax(-1) + 1
        logits_sorted = np.sort(logits_model_main, axis=-1)
        BvSB_model_main = logits_sorted[..., -1] - logits_sorted[..., -2]
        torch.cuda.empty_cache()

        tar_test, pre_test = tar_image[tar_image > 0], pre_model_main[tar_image > 0]
        OA_test, AA_mean_test, Kappa_test, AA_test = output_metric(tar_test, pre_test)

        OA_main_epoches.append(OA_test)
        AA_main_epoches.append(AA_mean_test)
        Kappa_main_epoches.append(Kappa_test)
        cmat = confusion_matrix(tar_test, pre_test)
        Recall_main_epoches.append(np.diagonal(cmat) / cmat.sum(1))
        Precision_main_epoches.append(np.diagonal(cmat) / (cmat.sum(0) + 1e-8))


    best_epoch = np.argmax(AA_main_epoches)
    print(f'Best epoch: OA: {OA_main_epoches[best_epoch]}, '
          f'AA: {AA_main_epoches[best_epoch]}, '
          f'Kappa: {Kappa_main_epoches[best_epoch]}')
    results = pd.DataFrame([OA_main_epoches[best_epoch],
                        AA_main_epoches[best_epoch],
                        Kappa_main_epoches[best_epoch]])
    return results



