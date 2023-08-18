# utils.py
# coding = utf-8

import tqdm
import numpy as np
import torch
import copy
import scipy.io as io
import torch.nn.functional as F
import torch.nn as nn
import cv2
from sklearn.metrics import accuracy_score, f1_score
from utils.loss import orthogonality_loss, similarity_loss, regularization

_CV2_BGR2HSV = cv2.COLOR_BGR2HSV
# get the hash from img_model


def get_data(opt, model, usable_gpu, TRAIN):
    v_preds, v_labels = [], []
    t_preds, t_labels = [], []
    valid_loss = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    if TRAIN:
            for i, (img1, img2, labels, index) in enumerate(opt.VALID_LOADER):
                if usable_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    labels = labels.cuda()
                preds, all_out, [shared_x, shared_y, exclusive_x, exclusive_y] = model(img1, img2)

                labels = labels.squeeze(dim=1)
                l_ort1 = orthogonality_loss(shared_x, exclusive_x)
                l_ort2 = orthogonality_loss(shared_y, exclusive_y)
                l_sim = similarity_loss(shared_x, shared_y)
                l_reg = regularization(all_out)

                l_cro = criterion(preds, labels.long())

                loss = l_cro + opt.ALPHA * (l_ort1 + l_ort2) + opt.BETA * l_sim - opt.DETA * l_reg
                preds = preds.max(1)[1].cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                length = len(preds)
                for k in range(length):
                    v_preds.append(preds[k])
                    v_labels.append(labels[k])

                valid_loss += loss.item() * len(img1)

                # fea = {key: fea[key].cuda() for key in fea}
                # value = value.cuda()
            valid_loss /= len(v_preds)
            v_acc = 100 * accuracy_score(v_labels, v_preds)
            v_f1 = 100 * f1_score(v_labels, v_preds)
            return v_acc, v_f1
    else:
        for i, (img1, img2, labels, index) in enumerate(opt.TEST_LOADER):
            if usable_gpu:
                img1 = img1.cuda()
                img2 = img2.cuda()
                labels = labels.cuda()
                # fea = {key: fea[key].cuda() for key in fea}
                # value = value.cuda()

            preds, all_out, [shared_x, shared_y, exclusive_x, exclusive_y] = model(img1, img2)

            labels = labels.squeeze(dim=1)
            l_ort1 = orthogonality_loss(shared_x, exclusive_x)
            l_ort2 = orthogonality_loss(shared_y, exclusive_y)
            l_sim = similarity_loss(shared_x, shared_y)
            l_reg = regularization(all_out)

            l_cro = criterion(preds, labels.long())

            loss = l_cro + opt.ALPHA * (l_ort1 + l_ort2) + opt.BETA * l_sim - opt.DETA * l_reg
            preds = preds.max(1)[1].cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            length = len(preds)
            for k in range(length):
                t_preds.append(preds[k])
                t_labels.append(labels[k])

            valid_loss += loss.item() * len(img1)

        test_loss /= len(v_preds)
        t_acc = 100 * accuracy_score(v_labels, v_preds)
        t_f1 = 100 * f1_score(v_labels, v_preds)
        return t_acc, t_f1



def resize_img(img):
    dim = (64, 64) # W:64,H:64

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
