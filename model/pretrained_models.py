import torch
import torch.nn as nn

from resnet50_face_sfew_dag import Resnet50_face_sfew_dag


class Resnet50_FER(nn.Module):
    def __init__(self, weights_path):
        super(Resnet50_FER, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        self.model.load_state_dict(torch.load(weights_path))

    def forward(self, imgs):
        out = self.model(imgs)
        return out
