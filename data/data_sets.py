# data_sets.py
# coding = utf-8

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import cv2
import torch.utils.data as data
_CV2_BGR2HSV = cv2.COLOR_BGR2HSV

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class _Phototriage(data.Dataset):
    type_gamma = 0
    type_train = 1
    type_test = 2
    type_train1 = 3
    base_folder = 'phototriage/'
    data_folder = 'train_val_imgs/'
    train_list = 'train.txt'
    test_list = 'test.txt'
    valid_list = 'valid.txt'
    valid_label_list = 'valid_ranks.txt'

    def __init__(self, root, type=1, transform=None, target_transform=None):
        train_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform1 = train_transform1
        self.transform2 = test_transform1
        self.target_transform = target_transform
        self.type = type  # training set or test set
        # self.path = self.root + self.base_folder + self.data_folder
        self.path = self.root + self.base_folder2 + self.data_folder



        # now load the picked numpy arrays
        if self.type < 2:
            self.train_data1 = []
            self.train_data2 = []
            self.train_labels = []
            fo = open(self.root + self.base_folder + self.train_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                # if float(temp[2]) < 0.3 or float(temp[2]) > 0.7:
                #     self.train_data1.append(np.array([temp[0]]))
                #     self.train_data2.append(np.array([temp[1]]))
                #     # self.train_labels.append(np.array([temp[2]]))
                #     # self.train_value.append(np.array([temp[2]]))
                #     self.train_labels.append(np.array([[int(la) for la in temp[3]]]))
                self.train_data1.append(np.array([temp[0]]))
                self.train_data2.append(np.array([temp[1]]))
                # self.train_labels.append(np.array([temp[2]]))
                # self.train_value.append(np.array([temp[2]]))
                self.train_labels.append(np.array([[int(la) for la in temp[3]]]))
            fo.close()

            self.train_data1 = np.concatenate(self.train_data1)
            self.train_data2 = np.concatenate(self.train_data2)
            # self.train_value = np.concatenate(self.train_value).astype(np.float32)
            # self.train_labels = np.concatenate(self.train_labels)
            self.train_labels = np.concatenate(self.train_labels).astype(np.float32)
            if self.type == self.type_gamma:
                self.gamma_data1 = self.train_data1
                self.gamma_data2 = self.train_data2
                self.gamma_labels = self.train_labels
                # self.gamma_value = self.train_value

        elif self.type == 2:
            self.test_data1 = []
            self.test_data2 = []
            self.test_labels = []
            # self.test_value = []
            fo = open(self.root + self.base_folder + self.test_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                if float(temp[2]) < 0.3 or float(temp[2]) > 0.7:
                    self.test_data1.append(np.array([temp[0]]))
                    self.test_data2.append(np.array([temp[1]]))
                    # self.test_labels.append(np.array([temp[2]]))
                    # self.test_value.append(np.array([temp[2]]))
                    self.test_labels.append(np.array([[int(la) for la in temp[3]]]))
            fo.close()

            self.test_data1 = np.concatenate(self.test_data1)
            self.test_data2 = np.concatenate(self.test_data2)
            self.test_labels = np.concatenate(self.test_labels).astype(np.float32)


        else:

            self.valid_data = []
            fo = open(self.root + self.base_folder + self.valid_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.valid_data.append(np.array([temp[0]]))
            fo.close()
            self.valid_data = np.concatenate(self.valid_data)
            self.valid_labels = []
            fo = open(self.root + self.base_folder + self.valid_label_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.valid_labels.append(np.array([[int(la) for la in temp[1:]]]))
            fo.close()

            self.valid_value = self.valid_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # fea = {'color1', 'color2', 'hsv1', 'hsv2', 'sift1', 'sift2', 'hog1', 'hog2'}
        fea = dict()

        if self.type == self.type_train:
            img_name1, img_name2, target = str(self.train_data1[index]), str(self.train_data2[index]), self.train_labels[index]
            img1 = self._loader(self.path + img_name1)
            img2 = self._loader(self.path + img_name2)
            # query1 = cv2.imread(self.path + img_name1)
            # query2 = cv2.imread(self.path + img_name2)
            # fea.update([('color1', color_feature(query1)), ('color2', color_feature(query2))])
            # fea.update([('hsv1', hsv_feature(query1)), ('hsv2', hsv_feature(query2))])
            # fea.update([('sift1', sift_feature(query1)), ('sift2', sift_feature(query2))])
            # fea.update([('hog1', hog_feature(query1)), ('hog2', hog_feature(query2))])
        elif self.type == self.type_gamma:
            img_name1, img_name2, target = str(self.gamma_data1[index]), str(self.gamma_data2[index]), self.gamma_labels[index]
            img1 = self._loader(self.path + img_name1)
            img2 = self._loader(self.path + img_name2)
            # query1 = cv2.imread(self.path + img_name1)
            # query2 = cv2.imread(self.path + img_name2)
            # fea.update([('color1', color_feature(query1)), ('color2', color_feature(query2))])
            # fea.update([('hsv1', hsv_feature(query1)), ('hsv2', hsv_feature(query2))])
            # fea.update([('sift1', sift_feature(query1)), ('sift2', sift_feature(query2))])
            # fea.update([('hog1', hog_feature(query1)), ('hog2', hog_feature(query2))])
        elif self.type == self.type_test:
            img_name1, img_name2, target = str(self.test_data1[index]), str(self.test_data2[index]), self.test_labels[index]
            img1 = self._loader(self.path + img_name1)
            img2 = self._loader(self.path + img_name2)
            # query1 = cv2.imread(self.path + img_name1)
            # query2 = cv2.imread(self.path + img_name2)
            # fea.update([('color1', color_feature(query1)), ('color2', color_feature(query2))])
            # fea.update([('hsv1', hsv_feature(query1)), ('hsv2', hsv_feature(query2))])
            # fea.update([('sift1', sift_feature(query1)), ('sift2', sift_feature(query2))])
            # fea.update([('hog1', hog_feature(query1)), ('hog2', hog_feature(query2))])
        else:
            img_name, target = str(self.train1_data[index]), self.train1_labels[index], self.train1_value
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)
            # query1 = cv2.imread(self.path + img_name)
            # query2 = cv2.imread(self.path + img_name)
            # fea.update([('color1', color_feature(query1)), ('color2', color_feature(query2))])
            # fea.update([('hsv1', hsv_feature(query1)), ('hsv2', hsv_feature(query2))])
            # fea.update([('sift1', sift_feature(query1)), ('sift2', sift_feature(query2))])
            # fea.update([('hog1', hog_feature(query1)), ('hog2', hog_feature(query2))])
        if self.transform is not None:
            if self.type <= 2:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img1, img2, target, index

    def _loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img1:
                return img1.convert('RGB')


    def setindex(self, index):
        if index is None:
            raise ValueError('index should not be None!')
        if self.type == self.type_gamma:
            self.gamma_data1 = self.train_data1[index]
            self.gamma_data2 = self.train_data2[index]
            self.gamma_labels = self.train_labels[index]
        else:
            raise TypeError('the type of data is not gamma!')

    def __len__(self):
        if self.type == self.type_train:
            return len(self.train_data1)
        elif self.type == self.type_gamma:
            return len(self.gamma_data1)
        elif self.type == self.type_test:
            return len(self.test_data1)
        else:
            return len(self.valid_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.type == self.type_train else 'test' if self.type == self.type_test else 'gamma'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class _Phototriage_Pre(data.Dataset):
    type_gamma = 0
    type_train = 1
    type_test = 2
    type_train_s = 3
    type_test_s = 4
    base_folder = 'rebuild_1/'
    base_folder2 = 'phototriage/'
    data_folder = 'train_val_imgs/'
    train_list = 'train_images_label.txt'
    test_list = 'valid_images_label.txt'
    train_s_list = 'train_series_label.txt'
    test_s_list = 'valid_series_label.txt'


    def __init__(self, root, type=1, transform=None, target_transform=None):
        train_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


        self.root = os.path.expanduser(root)
        self.transform = transform
        self.transform1 = train_transform1
        self.transform2 = test_transform1
        self.target_transform = target_transform
        self.type = type  # training set or test set
        self.path = self.root + self.base_folder2 + self.data_folder

        # now load the picked numpy arrays
        if self.type < 2:
            self.train_data = []
            self.train_labels = []
            fo = open(self.root + self.base_folder + self.train_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.train_data.append(np.array([temp[0]]))
                self.train_labels.append(np.array([[int(la) for la in temp[1:]]]))
            fo.close()
            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels).astype(np.float32)
            if self.type == self.type_gamma:
                self.gamma_data = self.train_data
                self.gamma_labels = self.train_labels

        elif self.type == 2:
            self.test_data = []
            self.test_labels = []
            fo = open(self.root + self.base_folder + self.test_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.test_data.append(np.array([temp[0]]))
                self.test_labels.append(np.array([[int(la) for la in temp[1:]]]))
            fo.close()
            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels).astype(np.float32)

        elif self.type == 3:
            self.train_data = []
            self.train_labels = []
            fo = open(self.root + self.base_folder + self.train_s_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.train_data.append(np.array([temp[0]]))
                self.train_labels.append(np.array([[int(la) for la in temp[1:]]]))
            fo.close()
            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels).astype(np.float32)

        # elif self.type == 4:
        else:
            self.test_data = []
            self.test_labels = []
            fo = open(self.root + self.base_folder + self.test_s_list)
            imgs = fo.readlines()
            for val in imgs:
                temp = val.split()
                self.test_data.append(np.array([temp[0]]))
                self.test_labels.append(np.array([[int(la) for la in temp[1:]]]))
            fo.close()
            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels).astype(np.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.type == self.type_train:
            img_name, target = str(self.train_data[index]), self.train_labels[index]
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)
        elif self.type == self.type_gamma:
            img_name, target = str(self.gamma_data[index]), self.gamma_labels[index]
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)
        elif self.type == self.type_test:
            img_name, target = str(self.test_data[index]), self.test_labels[index]
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)
        elif self.type == self.type_train_s:
            img_name, target = str(self.train_data[index]), self.train_labels[index]
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)
        # elif self.type == self.type_test_s:
        else:
            img_name, target = str(self.test_data[index]), self.test_labels[index]
            img = self._loader(self.path + img_name)
            img1 = self.transform1(img)
            img2 = self.transform1(img)

        if self.transform is not None:
            if self.type <= 2:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, index

    def _loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img1:
                return img1.convert('RGB')

    def setindex(self, index):
        if index is None:
            raise ValueError('index should not be None!')
        if self.type == self.type_gamma:
            self.gamma_data = self.train_data[index]
            self.gamma_labels = self.train_labels[index]
        else:
            raise TypeError('the type of data is not gamma!')

    def __len__(self):
        if self.type == self.type_train:
            return len(self.train_data)
        elif self.type == self.type_gamma:
            return len(self.gamma_data)
        elif self.type == self.type_test:
            return len(self.test_data)
        elif self.type == self.type_train_s:
            return len(self.train_data)
        elif self.type == self.type_test_s:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.type == self.type_train else 'test' if self.type == self.type_test else 'gamma'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    SIDE_LENGTH = 224
    transform = transforms.Compose([
        transforms.Resize(SIDE_LENGTH, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    TRAIN_DATA = _NUS(root='../', train=True, transform=transform, download=False)

    # TRAIN_LOADER = DataLoader(dataset=TRAIN_DATA, batch_size=16, shuffle=True, drop_last=True)
    TRAIN_LOADER = DataLoader(dataset=TRAIN_DATA, batch_size=16, shuffle=True, drop_last=True)
    for steps, (inputs, labels, index) in enumerate(TRAIN_LOADER):
        print(inputs.shape)
        print(labels.shape)
        print(TRAIN_DATA[0][0].shape)
        print(inputs[0])
        break
