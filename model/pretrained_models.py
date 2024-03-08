import torch
import torch.nn as nn

from model.resnet50_face_sfew_dag import Resnet50_face_sfew_dag


class Resnet50_FER(nn.Module):
    def __init__(self, weights_path, num_classes):
        super(Resnet50_FER, self).__init__()
        self.model = Resnet50_face_sfew_dag(num_classes)
        self.model.load_state_dict(torch.load(weights_path))

    def forward(self, imgs):
        out = self.model(imgs)
        return out
