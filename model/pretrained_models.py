import torch
import torch.nn as nn

from resnet50_face_sfew_dag import Resnet50_face_sfew_dag


class Resnet50_FER(nn.Module):
    def __init__(self, weights_path):
        super(Resnet50_FER, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.prediction = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, imgs):
        out = self.model(imgs)
        return out
