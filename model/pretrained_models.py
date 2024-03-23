import torch
import torch.nn as nn

from resnet50_face_sfew_dag import Resnet50_face_sfew_dag


class Resnet50_FER(nn.Module):
    def __init__(self, weights_path):
        super(Resnet50_FER, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        self.model.load_state_dict(torch.load(weights_path))
        # self.model.conv1_conv = nn.Conv2d(414, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3))
        self.model.prediction = nn.Linear(in_features=2048, out_features=1, bias=True)
        # self.model.prediction_avg = nn.AvgPool2d(kernel_size=1, stride=[1, 1], padding=0)

    def forward(self, imgs):
        out = self.model(imgs)
        return out


class Resnet50_FER_V2(nn.Module):
    def __init__(self, weights_path):
        super(Resnet50_FER_V2, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        loaded = self.model.load_state_dict(torch.load(weights_path))
        print(f"is loaded : {loaded}")
        self.model.prediction = nn.Linear(in_features=2048, out_features=5, bias=True)
        self.last = nn.Conv1d(kernel_size=5, in_channels=2048, out_channels=5)

    def forward(self, imgs):
        out = self.model(imgs)
        out = out.view(out.shape[0] // 5, out.shape[-1], 5)
        out = self.last(out).squeeze()

        return out
