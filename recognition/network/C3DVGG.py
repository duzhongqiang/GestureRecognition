import torch
import torch.nn as nn
from torchsummary import summary

class C3DVGG(nn.Module):
    """
    The C3DVGG network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3DVGG, self).__init__()

        #C3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        #VGG
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #2
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #4
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #6
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #7
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #9
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #10
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #12
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            #13
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            )
        self.classifier = nn.Sequential(
            torch.nn.Linear(41472, 4096),
            # torch.nn.Linear(29184, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, num_classes),
            )

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, y):
        # C3D
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        # print(x.shape)
        # xs = x.view(x.size(0), -1)
        # print(xs.shape)

        #VGG
        y = self.features(y) 
        # print(y.shape)
        # ys = y.view(y.size(0), -1)
        # print(ys.shape)
        combined = torch.cat((x.view(x.size(0), -1),
                          y.view(y.size(0), -1)), dim=1)
        
        logits = self.classifier(combined)
        # print(logits.shape)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_C3D_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        }
        
        corresp_VGG16_name = {
                        # Conv1
                        "features.0.weight": "features.0.weight",
                        "features.0.bias": "features.0.bias",
                        # Conv2
                        "features.2.weight": "features.2.weight",
                        "features.2.bias": "features.2.bias",
                        # Conv3
                        "features.5.weight": "features.5.weight",
                        "features.5.bias": "features.5.bias",
                        # Conv4
                        "features.7.weight": "features.7.weight",
                        "features.7.bias": "features.7.bias",
                        # Conv5
                        "features.10.weight": "features.10.weight",
                        "features.10.bias": "features.10.bias",
                        # Conv6
                        "features.12.weight": "features.12.weight",
                        "features.12.bias": "features.12.bias",
                        # Conv7
                        "features.14.weight": "features.14.weight",
                        "features.14.bias": "features.14.bias",
                        # Conv8
                        "features.17.weight": "features.17.weight",
                        "features.17.bias": "features.17.bias",
                        # Conv9
                        "features.19.weight": "features.19.weight",
                        "features.19.bias": "features.19.bias",
                        # Conv10
                        "features.21.weight": "features.21.weight",
                        "features.21.bias": "features.21.bias",
                        # Conv11
                        "features.24.weight": "features.24.weight",
                        "features.24.bias": "features.24.bias",
                        # Conv12
                        "features.26.weight": "features.26.weight",
                        "features.26.bias": "features.26.bias",
                        # Conv13
                        "features.28.weight": "features.28.weight",
                        "features.28.bias": "features.28.bias",
                        }


        C3D_dict = torch.load('./pretrainedModel/c3d-pretrained.pth')
        VGG16_dict = torch.load('./pretrainedModel/vgg16-397923af.pth')
        s_dict = self.state_dict() #获取当前网络参数
        for name in C3D_dict:
            if name not in corresp_C3D_name:
                continue
            s_dict[corresp_C3D_name[name]] = C3D_dict[name]
        for name in VGG16_dict:
            if name not in corresp_VGG16_name:
                continue
            s_dict[corresp_VGG16_name[name]] = VGG16_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    a = model.features[2]
    # b = model.features[0]
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.features[0], model.features[2], model.features[5],
         model.features[7],model.features[10],model.features[12],model.features[14],
         model.features[17],model.features[19],model.features[21],model.features[24],
         model.features[26],model.features[28]]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.classifier[0], model.classifier[3], model.classifier[6]]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    # inputs = torch.rand(1, 3, 16, 112, 112)
    x = torch.rand(1, 3, 32, 112, 112)
    y = torch.rand(1, 3, 224, 224)
    net = C3DVGG(num_classes=10, pretrained=True)
    # summary(net, (3, 32, 112, 112),device='cpu')

    outputs = net.forward(x,y)
    print(outputs.size())