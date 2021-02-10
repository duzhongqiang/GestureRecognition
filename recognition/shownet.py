import torch
import tensorwatch as tw
from network import C3DVGG

num_classes = 10

model = C3DVGG.C3DVGG(num_classes=num_classes)
# x = torch.rand(1, 3, 32, 112, 112)
# y = torch.rand(1, 3, 224, 224)
tw.draw_model(model, [1, 3, 32,512, 256], [1, 3, 224, 224])