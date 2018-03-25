# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   create time: 2018.03.14 Wed. 23h45m16s
   author: Chuanfeng Liu
   e-mail: microlj@126.com
   github: https://github.com/Knownx
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import cv2
import os

class Dataset(object):
    def __init__(self, all_data, numClasses, mean=np.array([104, 117, 124]),shuffle=True, trainPercent=0.7):
        self.all_data = all_data
        self.numClasses = numClasses
        self.shuffle = shuffle
        self.trainPercent = trainPercent
        if not os.path.exists('train.txt') & os.path.exists('validation.txt'):
            self.dataSplit(self.all_data, shuffle=self.shuffle, trainPercent=self.trainPercent)
        self.dataProcess()

        self.train_ptr = 0
        self.validation_ptr = 0
        self.mean = mean
        self.crop_size = 227

    # Return next batch with size batch_size for training or validation
    def getNextBatch(self, batch_size, phase):
        if phase == 'training':
            start = self.train_ptr
            end = min(start+batch_size, self.train_length)
            paths = self.train_image[start:end]
            labels = self.train_labels[start:end]
            self.train_ptr += batch_size

        elif phase == 'validation':
            start = self.validation_ptr
            end = min(start+batch_size, self.validation_length)
            paths = self.val_image[start:end]
            labels = self.val_labels[start:end]
            self.validation_ptr += batch_size
        else:
            return None, None

        # Process images into input size
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i, path in enumerate(paths):
            img = cv2.imread(path)
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            img = img.astype(np.float32)
            img -= self.mean
            images[i] = img

        # one-hot encode for labels
        one_hot_labels = np.zeros((batch_size, self.numClasses))
        for i, label in enumerate(labels):
            one_hot_labels[i][label] = 1

        return images, one_hot_labels

    # Split the input train.txt to train and validation sets
    def dataSplit(self, alldata, shuffle=True, trainPercent=0.7):
        with open(alldata, 'r') as f:
            f_train = open('train.txt', 'w')
            f_val = open('validation.txt', 'w')
            lines = f.readlines()
            numLines = len(lines)
            splitPoint = np.floor(numLines * trainPercent)
            if shuffle:
                permutation = np.random.permutation(numLines)
            else:
                permutation = range(numLines)
            curr_line = 0
            for i in permutation:
                if curr_line < splitPoint:
                    f_train.write(lines[i])
                else:
                    f_val.write(lines[i])
                curr_line += 1
            f_train.close()
            f_val.close()

    def dataProcess(self):
        # Load train images and labels
        with open('train.txt','r') as f:
            lines = f.readlines()
            self.train_length = len(lines)
            self.train_image = []
            self.train_labels = []
            for line in lines:
                items = line.strip().split()
                self.train_image.append(items[0])
                self.train_labels.append(int(items[1]))
        f.close()

        # Load validation images and labels
        with open('validation.txt','r') as f:
            lines = f.readlines()
            self.validation_length=len(lines)
            self.val_image = []
            self.val_labels = []
            for line in lines:
                items = line.strip().split()
                self.val_image.append(items[0])
                self.val_labels.append(int(items[1]))
        f.close()

    def reset_ptr(self):
        self.train_ptr = 0
        self.validation_ptr = 0



