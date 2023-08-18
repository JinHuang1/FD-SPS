# main_circle.py
# encoding = utf-8

import torch
import os
import time

import torch.nn as nn
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import numpy as np
from datetime import datetime
from configs import MethodConfig
from utils import get_data
from utils.loss import orthogonality_loss, similarity_loss, regularization
from models.FDSPS import FDSPS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set the gpu


def train(opt):
    # initialize the net
    # judge whether using 'extend' mode and whether the temp file is existed
    if opt.EXTEND and os.access(opt.TEMP_SAVE_NAME, os.F_OK):

        temp_variances = torch.load(opt.TEMP_SAVE_NAME)

        model = FDSPS()
        model.load_state_dict(temp_variances['model'])
        optimizer_shared_encoder = torch.optim.SGD(model.sh_enc.parameters(), lr=opt.LR, weight_decay=1e-4)
        optimizer_shared_encoder.load_state_dict(temp_variances['optimizer_shared_encoder'])

        optimizer_exclusive_encoder = torch.optim.SGD(model.ex_enc.parameters(), lr=opt.LR, weight_decay=1e-4)
        optimizer_exclusive_encoder.load_state_dict(temp_variances['optimizer_exclusive_encoder'])

        optimizer_classifier = torch.optim.SGD(model.classifier.parameters(), lr=opt.LR, weight_decay=1e-4)
        optimizer_classifier.load_state_dict(temp_variances['optimizer_classifier'])

        scheduler_shared_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_shared_encoder, opt.MILESTONES,
                                                                gamma=opt.LR_DECAY_RATE)
        scheduler_shared_encoder.load_state_dict(temp_variances['scheduler_shared_encoder'])

        scheduler_exclusive_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_exclusive_encoder,
                                                                   opt.MILESTONES, gamma=opt.LR_DECAY_RATE)
        scheduler_exclusive_encoder.load_state_dict(temp_variances['scheduler_exclusive_encoder'])

        scheduler_classifier = torch.optim.lr_scheduler.MultiStepLR(optimizer_classifier, opt.MILESTONES,
                                                            gamma=opt.LR_DECAY_RATE)
        scheduler_classifier.load_state_dict(temp_variances['scheduler_classifier'])

        start = temp_variances['epoch']
    else:
        model = FDSPS()
        # Network optimizers
        optimizer_shared_encoder = torch.optim.SGD(model.sh_enc.parameters(), lr=opt.LR, weight_decay=1e-4)
        optimizer_exclusive_encoder = torch.optim.SGD(model.ex_enc.parameters(), lr=opt.LR, weight_decay=1e-4)
        optimizer_classifier = torch.optim.SGD(model.classifier.parameters(), lr=opt.LR, weight_decay=1e-4)
        scheduler_shared_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_shared_encoder, opt.MILESTONES,
                                                                        gamma=opt.LR_DECAY_RATE)
        scheduler_exclusive_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_exclusive_encoder,
                                                                           opt.MILESTONES, gamma=opt.LR_DECAY_RATE)
        scheduler_classifier = torch.optim.lr_scheduler.MultiStepLR(optimizer_classifier, opt.MILESTONES,
                                                                    gamma=opt.LR_DECAY_RATE)
        start = 0

    print('GPU-ID:', os.environ["CUDA_VISIBLE_DEVICES"])
    initial_time = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time()))
    log_file = open(opt.LOG_SAVE_NAME, 'a')
    log_file.write('Training Begin!\nTime: ' + initial_time + '\n')
    log_file.flush()
    log_file_best = open(opt.LOG_BEST_SAVE_NAME, 'a')

    model.train()
    _zero = torch.Tensor([0])
    usable_gpu = opt.USE_GPU and torch.cuda.is_available()
    if usable_gpu:
        model = model.cuda()
        _zero = _zero.cuda()
        opt.TRAIN_LABELS = opt.TRAIN_LABELS.cuda()

    acces = [-np.inf]
    criterion = nn.CrossEntropyLoss()
    cuda = True if torch.cuda.is_available() else False

    for epoch in range(start, opt.EPOCH):
        train_loss = 0
        ep_time = datetime.now()
        t_preds, t_labels = [], []

        for steps, (inputs1, inputs2, labels, index) in enumerate(opt.TRAIN_LOADER):
            _labels = opt.TRAIN_LABELS[index]
            if usable_gpu:
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
                labels = labels.cuda()

            labels = labels.squeeze(dim=1)
            preds, all_out, [shared_x, shared_y, exclusive_x, exclusive_y] = model(inputs1, inputs2)

            l_ort1 = orthogonality_loss(shared_x, exclusive_x)
            l_ort2 = orthogonality_loss(shared_y, exclusive_y)
            l_sim = similarity_loss(shared_x, shared_y)
            l_reg = regularization(all_out)
            l_cro = criterion(preds, labels.long())

            loss = l_cro + opt.ALPHA * (l_ort1 + l_ort2) + opt.BETA * l_sim + opt.DETA * l_reg

            print("[Train] [l_ort1 %f] [l_ort2 %f] [l_sim %f] [l_reg %f] [l_cro %f]" % (
                (opt.ALPHA * l_ort1), (opt.ALPHA * l_ort2), (opt.BETA * l_sim), (opt.DETA * l_reg), l_cro))
            print("[Train] [Epoch %d/%d][step %d/%d] [Batch Loss: %f]" % (
            epoch + 1, opt.EPOCH, steps, len(opt.TRAIN_DATA) / opt.BATCH_SIZE, loss))

            """Set all the networks gradient to zero"""
            optimizer_shared_encoder.zero_grad()
            optimizer_exclusive_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.CLIP)

            optimizer_shared_encoder.step()
            optimizer_exclusive_encoder.step()
            optimizer_classifier.step()

            preds = preds.max(1)[1].cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            length = len(preds)
            for k in range(length):
                t_preds.append(preds[k])
                t_labels.append(labels[k])
            # total_loss += J.item()
            # counting training loss
            train_loss += loss.item() * len(inputs1)

        train_loss /= len(t_preds)
        t_acc = 100 * accuracy_score(t_labels, t_preds)
        t_f1 = 100 * f1_score(t_labels, t_preds)
        print("[Train] [Epoch %d/%d] [Accuracy %f] [Epoch Loss: %f]" % (epoch+1, opt.EPOCH, t_acc, train_loss))
        print("[Train] [Epoch %d/%d] [F1_Score %f] [Epoch Loss: %f]" % (epoch + 1, opt.EPOCH, t_f1, train_loss))

        print('=================time cost: {}==================='.format(datetime.now() - ep_time))

        v_preds, v_labels = [], []
        valid_loss = 0
        for i, (img1, img2, labels, index) in enumerate(opt.TEST_LOADER):
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

            loss = l_cro + opt.ALPHA * (l_ort1 + l_ort2) + opt.BETA * l_sim + opt.DETA * l_reg

            preds = preds.max(1)[1].cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            length = len(preds)
            for k in range(length):
                v_preds.append(preds[k])
                v_labels.append(labels[k])

            valid_loss += loss.item() * len(img1)

        valid_loss /= len(v_preds)
        v_acc = 100 * accuracy_score(v_labels, v_preds)
        v_f1 = 100 * f1_score(v_labels, v_preds)

        # scheduler.step()  # update the learning rate
        scheduler_shared_encoder.step()
        scheduler_exclusive_encoder.step()
        scheduler_classifier.step()
        print('Epoch:', epoch + 1, '| Acc =', v_acc)
        print('Epoch:', epoch + 1, '| F1 Score =', v_f1)
        print('Epoch:', epoch + 1, '| test_loss =', valid_loss)
        log_file.write('Epoch: ' + str(epoch + 1) + ' | loss = ' + str(valid_loss) + '\n')
        log_file.write('Epoch: ' + str(epoch + 1) + ' | acc = ' + str(v_acc) + '\n')
        log_file.write('Epoch: ' + str(epoch + 1) + ' | f1 = ' + str(v_f1) + '\n')
        log_file.flush()

        # save temp variances
        temp_variances = ({
            'epoch': epoch + 1,
            'net': model.state_dict(),
            'optimizer_shared_encoder': optimizer_shared_encoder.state_dict(),
            'optimizer_exclusive_encoder': optimizer_exclusive_encoder.state_dict(),
            'optimizer_classifier': optimizer_classifier.state_dict(),

            'scheduler_shared_encoder': scheduler_shared_encoder.state_dict(),
            'scheduler_exclusive_encoder': scheduler_exclusive_encoder.state_dict(),
            'scheduler_classifier': scheduler_classifier.state_dict(),

        })
        torch.save(temp_variances, opt.TEMP_SAVE_NAME)
        torch.cuda.empty_cache()

    if v_acc > np.max(acces):
        # save model at each iter
        # torch.save(UrbanFM.state_dict(),
        #            '{}/model-{}.pt'.format(save_path, iter))
        print('Save model!')
        torch.save(model.state_dict(), opt.NET_SAVE_NAME)
        log_file_best.write('Epoch: ' + str(epoch + 1) + ' | loss = ' + str(valid_loss) + '\n')
        log_file_best.write('Epoch: ' + str(epoch + 1) + ' | acc = ' + str(v_acc) + '\n')
        log_file_best.write('Epoch: ' + str(epoch + 1) + ' | acc = ' + str(v_f1) + '\n')
        log_file_best.flush()
        log_file_best.close()

    print("[valid] [Epoch %d/%d] [Accuracy %f] [F1_score %f] [Batch Loss: %f]" % (epoch+1, opt.EPOCH, v_acc, v_f1, valid_loss))
    acces.append(v_acc)

    log_file.write('Training End!\n\n')
    log_file.close()

    # torch.save(net.state_dict(), opt.NET_SAVE_NAME)
    print('=================time cost: {}==================='.format(datetime.now() - ep_time))
    return model, usable_gpu


def evaluate(opt):
    model = FDSPS()
    model.load_state_dict(torch.load(opt.NET_SAVE_NAME))
    usable_gpu = opt.USE_GPU and torch.cuda.is_available()
    if usable_gpu:
        model.cuda()
    return model, usable_gpu


def compute(opt, model, usable_gpu, TRAIN):
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    cuda = True if torch.cuda.is_available() else False

    v_acc, v_f1 = get_data(opt, model, usable_gpu, TRAIN)

    initial_time = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time()))
    log_file = open(opt.LOG_SAVE_NAME, 'a')
    log_file.write('computeAcc Result!\nTime: ' + initial_time + '\nAcc: ' + str(v_acc) + '\n\n')
    log_file.close()
    return v_acc, v_f1


if __name__ == '__main__':
    # dataset, {0: phototriage, 1: ……}
    data_selected = 0
    lr = 7e-2 # learning rate
    batch_size = 16
    extend = False  # whether continue train samples from the last training
    TRAIN = True  # whether train samples with mutual information
    # TRAIN = False
    alpha = 0.02
    beta = 0.8
    deta = 0.002

    if data_selected == 0: data_name = MethodConfig.Phototriage

    net_save_name = './results/' + data_name + '/FDSPS' +'.pkl'
    temp_save_name = './temp/' + data_name + '/FDSPS' + '.pkl'
    log_save_name = './log/' + data_name + '/FDSPS' + '.txt'
    log_best_save_name = './log/' + data_name + '/best_FDSPS' + '.txt'
    data_path = './Phototriage/data/'

    opt = MethodConfig(data_name=data_name, data_path=data_path,
                       log_best_save_name=log_best_save_name, net_save_name=net_save_name,
                       temp_save_name=temp_save_name, log_save_name=log_save_name,
                       extend=extend, batch_size=batch_size, lr=lr,
                       alpha=alpha, beta=beta, deta=deta)

    print('*** dataset:', data_name, '***')
    print('*** net_save_name:', net_save_name, '***')
    print('*** temp_save_name:', temp_save_name, '***')
    print('*** log_save_name:', log_save_name, '***')
    print('*** log_best_save_name:', log_best_save_name, '***')

    if TRAIN:
        model, use_gpu = train(opt)
        with torch.no_grad():
            v_acc, v_f1 = compute(opt, model, use_gpu, TRAIN)
        print(v_acc)
        print(v_f1)
    else:
        model, use_gpu = evaluate(opt)
        with torch.no_grad():
            v_acc, v_f1 = compute(opt, model, use_gpu, TRAIN)

        print(v_acc)
        print(v_f1)

