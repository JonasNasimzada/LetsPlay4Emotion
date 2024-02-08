import argparse

import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample
)
from pytorchvideo.transforms import (
    RandomShortSideScale

)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Precision, Recall
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import (
    RandomCrop,
    RandomHorizontalFlip
)
from torchvision.transforms._transforms_video import (
    NormalizeVideo
)
from torchvision.transforms.v2 import RandomAffine, GaussianBlur


class NeuralNetworkModel(LightningModule):
    def __init__(self, num_classes, train_dataset_file, val_dataset_file, nn_model):
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.val_dataset_file = val_dataset_file
        self.conv_layers = nn_model
        self.lr = 1e-3
        self.batch_size = 64
        self.num_worker = 8

        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.mean_squared_log_error = torchmetrics.MeanSquaredLogError()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.recall = Recall(task="multiclass", average='macro', num_classes=3)

        self.validation_output_list = []

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        train_dataset = labeled_video_dataset(self.train_dataset_file, clip_sampler=make_clip_sampler('uniform', 2),
                                              transform=video_transform, decode_audio=False)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker)
        return loader

    def _common_step(self, batch, batch_idx):
        video, input_label = batch['video'], batch['label']
        output_network = self.forward(video)
        loss = self.loss(output_network, input_label)
        return loss, output_network, input_label

    def training_step(self, batch, batch_idx):
        loss, output_network, input_label = self._common_step(batch, batch_idx)
        input_label_tensor = input_label.view(-1)

        self.log_dict(
            {
                "train_loss": loss,
                "train_f1_score": self.f1_score(output_network, input_label_tensor),
                "train_accuracy": self.accuracy(output_network, input_label_tensor),
                "train_precision": self.precision(output_network, input_label_tensor),
                "train_mean_squared_log_error": self.mean_squared_log_error(output_network, input_label_tensor),
                "train_confusion_matrix": self.confusion_matrix(output_network, input_label_tensor),
                "train_auroc": self.auroc(output_network, input_label_tensor),
                "train_mean_absolute_error": self.mean_absolute_error(output_network, input_label_tensor),
                "train_mean_squared_error": self.mean_squared_error(output_network, input_label_tensor),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "output_network": output_network, "input_label": input_label}

    def val_dataloader(self):
        train_dataset = labeled_video_dataset(self.val_dataset_file, clip_sampler=make_clip_sampler('uniform', 2),
                                              transform=video_transform, decode_audio=False)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker)
        return loader

    def validation_step(self, batch, batch_idx):
        loss, output_network, input_label = self._common_step(batch, batch_idx)

        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pred = {"loss": loss, "output_network": output_network, "input_label": input_label}
        self.validation_output_list.append(pred)

        return pred

    def on_validation_epoch_end(self):
        output_network = torch.cat([x["output_network"] for x in self.validation_output_list])
        input_label = torch.cat([x["input_label"] for x in self.validation_output_list])
        input_label_tensor = input_label.view(-1)
        self.log_dict(
            {
                "val_f1_score": self.f1_score(output_network, input_label_tensor),
                "val_accuracy": self.accuracy(output_network, input_label_tensor),
                "val_precision": self.precision(output_network, input_label_tensor),
                "val_mean_squared_log_error": self.mean_squared_log_error(output_network, input_label_tensor),
                "val_confusion_matrix": self.confusion_matrix(output_network, input_label_tensor),
                "val_auroc": self.auroc(output_network, input_label_tensor),
                "val_mean_absolute_error": self.mean_absolute_error(output_network, input_label_tensor),
                "val_mean_squared_error": self.mean_squared_error(output_network, input_label_tensor),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_output_list.clear()

    def predict_step(self, batch, batch_idx, dataloader=0):
        video, input_label = batch['video'], batch['label']
        output_network = self.forward(video)
        prediction = torch.argmax(output_network, dim=1)
        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['train', 'pred'])
    parser.add_argument("--type", required=True, choices=['binary', 'multi'])
    parser.add_argument("--dataset_infix", default="", help='optional infix for choosing dataset')
    parser.add_argument("--version", default="", type=int)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--precision", default=32, type=int)
    args = parser.parse_args()

    data_suffix = args.type
    data_infix = ""
    classes = None
    version = "v1"

    if args.dataset_infix != "":
        data_infix = f"{args.dataset_infix}_"

    if args.version != "":
        version = f"v{args.version}_"

    if args.type == 'binary':
        classes = 2
    elif args.type == 'multi':
        classes = 5

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 25

    video_transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                RandomAffine(degrees=20, translate=(0, 0.1), shear=(-15, 15, -15, 15)),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                RandomShortSideScale(
                    min_size=256,
                    max_size=320,
                ),
                RandomCrop(256),
                RandomHorizontalFlip(p=0.5),
            ]
        ),
    )
    model_resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model_resnet.blocks[5].proj = nn.Linear(in_features=2048, out_features=classes, bias=True)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=f'checkpoints_{version}{data_suffix}',
                                          filename=f"ckpt-{version}{data_infix}{data_suffix}" + '-{epoch:02d}-{'
                                                                                                'val_loss:.2f}',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    strategy = DeepSpeedStrategy(logging_batch_size_per_gpu=64)

    logger_name = f"model_{version}{data_infix}{args.type}"
    logger = TensorBoardLogger("model_logger", name=logger_name)
    print(f"the logger name is: {logger_name}")

    torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        min_epochs=1,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_progress_bar=True,
        strategy=strategy,
        precision=args.precision,
        logger=logger
    )

    train_set = f"datasets/train-{data_infix}{data_suffix}.csv"
    val_set = f"datasets/val-{data_infix}{data_suffix}.csv"

    model = NeuralNetworkModel(
        num_classes=classes,
        train_dataset_file=train_set,
        val_dataset_file=val_set,
        nn_model=model_resnet,

    )
    print(f"this train set is gonna be used: {train_set}")
    print(f"this val set is gonna be used: {val_set}")

    if args.mode == 'train':
        trainer.fit(model)
    elif args.mode == 'pred':
        checkpoint_path = checkpoint_callback.best_model_path
        trainer.predict(model, checkpoint_path)
