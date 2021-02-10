from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import timeit
from datetime import datetime
import torch
from dataset import VideoDataset
from network import C3DVGG
from tqdm import tqdm
def test():
    num_classes = 10
    labels2 = dict()
    with open('./dataloaders/labels.txt','r') as f:
        for line in f.readlines():
            line = line.split()
            labels2[eval(line[0])] = line[1]
    f.close()
    # print(labels2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device being used:", device)
    model = C3DVGG.C3DVGG(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    model.to(device)
    criterion.to(device)
    model.eval()

    checkpoint = torch.load('./result/C3DVGG16_epoch-7.pth.tar',map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU 
    model.load_state_dict(checkpoint['state_dict'])

    test_dataloader  = DataLoader(VideoDataset(split='testone'), batch_size=1, num_workers=4)

    for RdData, AtmData, labels in test_dataloader:
        RdData = RdData.to(device)
        AtmData = AtmData.to(device)

        with torch.no_grad():
            outputs = model(RdData,AtmData)
        probs = nn.Softmax(dim=1)(outputs)
        preds = (torch.max(probs, 1)[1])
        preds = preds.data.cpu().numpy()
        label = labels2[preds[0]]
        print('\nThe Gesture prediction is ' + label)

if __name__ == "__main__":
    test()