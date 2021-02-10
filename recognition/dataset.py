import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, split='train'):

        self.RdSize = 112
        self.AtmSize = 224

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        if(split == 'train'):
            with open('./dataloaders/train.txt','r') as ftrain:
                for line in ftrain.readlines():
                    line = line.split()
                    self.fnames.append(line[0])
                    labels.append(line[1])
            ftrain.close()
        if(split == 'val'):
            with open('./dataloaders/valid.txt','r') as fvalid:
                for line in fvalid.readlines():
                    line = line.split()
                    self.fnames.append(line[0])
                    labels.append(line[1])
            fvalid.close()
        if(split == 'test'):
            with open('./dataloaders/test.txt','r') as ftest:
                for line in ftest.readlines():
                    line = line.split()
                    self.fnames.append(line[0])
                    labels.append(line[1])
            ftest.close()
        if(split == 'testone'):
            with open('./dataloaders/testone.txt','r') as ftestone:
                for line in ftestone.readlines():
                    line = line.split()
                    self.fnames.append(line[0])
                    labels.append(line[1])
            ftestone.close()

        assert len(labels) == len(self.fnames)
        print('Number of {} dataSets: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        if(split == 'train'):
            with open('./dataloaders/labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id) + ' ' + label + '\n')
            f.close()


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        RdData, AtmData = self.load_frames(self.fnames[index]) #load Data
        labels = np.array(self.label_array[index])

        RdData = self.normalize(RdData)
        AtmData = self.normalize(AtmData)
        RdData = self.Rdto_tensor(RdData)
        AtmData = self.Atmto_tensor(AtmData)
        return torch.from_numpy(RdData).type(torch.FloatTensor), torch.from_numpy(AtmData).type(torch.FloatTensor),torch.from_numpy(labels)

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame = frame / 255;
            buffer[i] = frame
        return buffer

    def Rdto_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))
    
    def Atmto_tensor(self, buffer):
        return buffer.transpose((2, 0, 1))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        assert frame_count == 33
        RdData = np.empty((frame_count - 1, self.RdSize, self.RdSize,3), np.dtype('float32'))
        AtmData = np.empty((self.AtmSize, self.AtmSize,3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            if i < (frame_count - 1):
                RdData[i] = frame
            elif(i == (frame_count - 1)):
                AtmData = frame

        return RdData, AtmData


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(split='train')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    print(enumerate(train_loader))

    for inputs_RD, inputs_ATM, labels in train_loader:
        print(inputs_RD)

    for i, sample in enumerate(train_loader):
        RdData = sample[0]
        AtmData = sample[1]
        labels = sample[2]
        print(RdData.size())
        print(AtmData.size())
        print(labels)

        if i == 1:
            break