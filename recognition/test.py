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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device being used:", device)
    model = C3DVGG.C3DVGG(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    model.to(device)
    criterion.to(device)
    model.eval()

    checkpoint = torch.load('./result/C3DVGG16_epoch-35.pth.tar',map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU 
    model.load_state_dict(checkpoint['state_dict'])

    test_dataloader  = DataLoader(VideoDataset(split='test'), batch_size=2, num_workers=4)
    test_size = len(test_dataloader.dataset)

    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0.0

    for RdData, AtmData, labels in tqdm(test_dataloader):
        RdData = RdData.to(device)
        AtmData = AtmData.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(RdData,AtmData)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels.long())

        running_loss += loss.item() * RdData.size(0)
        running_corrects += torch.sum(preds == labels.data.long())

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print("\nLoss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("\nExecution time: " + str(stop_time - start_time) + "\n")

if __name__ == "__main__":
    test()