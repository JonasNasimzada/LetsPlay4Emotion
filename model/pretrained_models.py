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
    def __init__(self, weights_path, mode):
        super(Resnet50_FER_V2, self).__init__()
        self.model = Resnet50_face_sfew_dag()
        loaded = self.model.load_state_dict(torch.load(weights_path))
        print(f"is loaded : {loaded}")
        out_weight_channel = 5
        if mode == "binary":
            out_weight_channel = 1

        self.model.prediction = nn.Linear(in_features=2048, out_features=5, bias=True)
        self.last = nn.Sequential(nn.Conv1d(kernel_size=5, in_channels=2048, out_channels=out_weight_channel),
                                  # mehrer 1D geschachetelt / ander kernel size, mit pooling layer/ attention über
                                  nn.AdaptiveMaxPool1d(output_size=1))  # Averagepooling

    def forward(self, imgs, batch_size):
        out = self.model(imgs)
        out = out.view(batch_size, -1, out.shape[0] // batch_size)

        out = self.last(out).squeeze()

        return out
