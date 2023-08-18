# method_config.py
# coding = utf-8

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import warnings
from data import _Phototriage


class MethodConfig(object):
    Phototriage = 'phototriage'

    SIDE_LENGTH = 224
    transform = transforms.Compose([
        transforms.Resize([SIDE_LENGTH, SIDE_LENGTH], Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, data_name, data_path, net_save_name='FDSPS.pkl', temp_save_name='temp_net.pkl',
                 log_best_save_name='log_best_net.txt', log_save_name='log_net.txt', log2_save_name='log_net.txt',
                 temperature=0.5, n_views=2, out_feat=512, nhid=64,
                 batch_size=64, epoch=2, lr=0.07, momentum=0, weight_decay=1e-5,
                 lr_decay_epoch=50, lr_decay_rate=0.1, milestones=[0.5, 0.7, 0.9], last_lr=None, use_gpu=True,
                 extend=False, model_ema=True, model_ema_decay=0.99996, model_ema_force_cpu=False,
                 alpha=0., beta=0., deta=0.):

        # save name for results of network
        self.MODEL_EMA = model_ema
        self.MODEL_EMA_DECAY = model_ema_decay
        self.MODEL_EMA_FORCE = model_ema_force_cpu
        self.NET_SAVE_NAME = net_save_name

        self.TEMP_SAVE_NAME = temp_save_name
        self.LOG_SAVE_NAME = log_save_name
        self.LOG_BEST_SAVE_NAME = log_best_save_name
        # self.MODE = mode
        self.EXTEND = extend
        # hyper-parameters
        # self.ETA = 0.02  # weight of quantization item
        self.CLIP = 1  # scale limitation of network gradient
        self.MARGIN = 0.4
        self.GAMMA = 1
        self.TAF = 1
        # parameters for network
        self.BATCH_SIZE = batch_size
        self.EPOCH = epoch
        self.LR = lr
        self.MOMENTUM = momentum
        self.WEIGHT_DECAY = weight_decay
        self.LR_DECAY_EPOCH = lr_decay_epoch
        self.MILESTONES = [int(self.EPOCH * milestone) for milestone in milestones]
        self.LR_DECAY_RATE = lr_decay_rate
        self.LAST_LR = last_lr
        self.TEMPERATURE = temperature
        self.N_VIEWS = n_views
        self.OUT_FEAT = out_feat
        self.NHID = nhid
        # parameter for gpu
        self.USE_GPU = use_gpu
        # data set
        self.DATA_PATH = data_path
        self.ALPHA = alpha
        self.BETA = beta
        self.DETA = deta

        self.DATA_PATH = data_path
        if type(data_name) != str:
            warnings.warn('Warning: data_name should be str type!')
        elif data_name.lower() == self.Phototriage:  # phototriage data set
            self.DATA_NAME = self.Phototriage
            # hyper-parameters

            # training set
            self.TRAIN_DATA = _Phototriage(root=self.DATA_PATH, type=_Phototriage.type_train, transform=self.transform)
            # test set
            self.TEST_DATA = _Phototriage(root=self.DATA_PATH, type=_Phototriage.type_test, transform=self.transform)
            # train1 set
            self.TRAIN1_DATA = _Phototriage(root=self.DATA_PATH, type=_Phototriage.type_train1,
                                            transform=self.transform)

            # parameters for data set
            self.NUM_CA = 10  # the number of categories
            self.LABEL_IS_SINGLE = True  # whether the label is single data
            self.NUM_WORKERS = 1
            # parameters for data set
            self.NUM_TR = self.TRAIN_DATA.__len__()  # the number of training set
            self.NUM_TE = self.TEST_DATA.__len__()  # the number of test set
            self.NUM_TR1 = self.TRAIN1_DATA.__len__()  # the number of training1 set

            self.TRAIN_LABELS = torch.Tensor(self.TRAIN_DATA.train_labels)

        self.TRAIN_LOADER = DataLoader(dataset=self.TRAIN_DATA, batch_size=self.BATCH_SIZE, shuffle=True,
                                       num_workers=self.NUM_WORKERS, drop_last=True)
        # train1 loader
        self.TRAIN1_LOADER = DataLoader(dataset=self.TRAIN_DATA, batch_size=self.BATCH_SIZE, shuffle=True,
                                        num_workers=self.NUM_WORKERS, drop_last=True)
        # test loader
        self.TEST_LOADER = DataLoader(dataset=self.TEST_DATA, batch_size=self.BATCH_SIZE, shuffle=False,
                                      num_workers=self.NUM_WORKERS, drop_last=True)

        self.NUM_TR = self.TRAIN_DATA.__len__()  # the number of training set
        self.NUM_TE = self.TEST_DATA.__len__()  # the number of test set


if __name__ == '__main__':
    opt = MethodConfig(MethodConfig.Phototriage, 'D:/PycharmProjects/data/')
    for steps, (inputs1, inputs2, labels, index) in enumerate(opt.TRAIN_LOADER):
        print(steps)
        print(inputs1.shape)
        print(inputs2.shape)
        print(labels.shape)
        print(index)
        break
