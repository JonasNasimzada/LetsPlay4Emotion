import argparse
import os

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample
)
from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix, AUROC
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import (
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip
)
from torchvision.transforms._transforms_video import (
    NormalizeVideo
)
from torchvision.transforms.v2 import RandomAffine


class NeuralNetworkModel(LightningModule):
    def __init__(self, num_classes, model_type, train_dataset_file, val_dataset_file, nn_model, augmentation_train,
                 augmentation_val):
        super().__init__()
        self.train_dataset_file = train_dataset_file
        self.val_dataset_file = val_dataset_file
        self.augmentation_train = augmentation_train
        self.augmentation_val = augmentation_val
        self.conv_layers = nn_model
        # Learning Rate Rumprobieren, 1e-4 ist mein standard auf 2d bildern und 3d Volumen
        self.lr = 1e-4
        # Kann man ein wenig noch hochschrauben - 8BS -> 32GB VRAM
        self.batch_size = 16
        self.num_worker = 8

        self.f1_score = F1Score(task=model_type, num_classes=num_classes)
        self.accuracy = Accuracy(task=model_type, num_classes=num_classes)
        self.auroc = AUROC(task=model_type, num_classes=num_classes)
        self.precision = Precision(task=model_type, average='macro', num_classes=num_classes)
        self.recall = Recall(task=model_type, average='macro', num_classes=num_classes)

        if model_type == "binary":
            self.confusion_matrix = ConfusionMatrix(task=model_type, num_classes=(++num_classes))
            self.loss = nn.BCELoss
        else:
            self.confusion_matrix = ConfusionMatrix(task=model_type, num_classes=num_classes)
            self.loss = nn.CrossEntropyLoss()

        self.validation_output_list = []

        # Wenn binary dann, https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    def forward(self, x):
        x = self.conv_layers(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        train_dataset = labeled_video_dataset(self.train_dataset_file, clip_sampler=make_clip_sampler('uniform', 2),
                                              transform=self.augmentation_train, decode_audio=False)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker)
        return loader

    def _common_step(self, batch, batch_idx):
        video, input_label = batch['video'], batch['label']
        input_label = input_label.to(torch.int64)
        output_network = self.forward(video)
        output_network = output_network
        loss = self.loss(output_network, input_label)
        return loss, output_network, input_label

    def training_step(self, batch, batch_idx):
        loss, output_network, input_label = self._common_step(batch, batch_idx)
        pred = {"loss": loss, "output_network": output_network, "input_label": input_label}

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return pred

    def val_dataloader(self):
        val_dataset = labeled_video_dataset(self.val_dataset_file, clip_sampler=make_clip_sampler('uniform', 2),
                                            transform=self.augmentation_val, decode_audio=False)
        loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker)
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
            sync_dist=True
        )
        pred = {"loss": loss, "output_network": output_network, "input_label": input_label}
        self.validation_output_list.append(pred)

        return pred

    def on_validation_epoch_end(self):
        output_network = torch.cat([x["output_network"] for x in self.validation_output_list])
        input_label = torch.cat([x["input_label"] for x in self.validation_output_list])

        self.confusion_matrix(output_network, input_label)
        self.f1_score(output_network, input_label)
        self.accuracy(output_network, input_label)
        self.precision(output_network, input_label)
        self.auroc(output_network, input_label)

        self.log_dict(
            {
                "val_f1_score": self.f1_score,
                "val_accuracy": self.accuracy,
                "val_precision": self.precision,
                "val_auroc": self.auroc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        confusion_matrix_computed = self.confusion_matrix.compute().detach().cpu().numpy().astype(int)
        df_cm = pd.DataFrame(confusion_matrix_computed, index=range(2), columns=range(2))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        self.logger.experiment.add_figure("val_confusion_matrix matrix", fig, self.current_epoch)

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
    parser.add_argument("--logger_comment", default="")
    parser.add_argument("--version", default="", type=int)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--precision", default=32, type=int)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    data_suffix = args.type
    data_infix = ""
    classes = None
    version = "v1"
    metric = None

    if args.dataset_infix != "":
        data_infix = f"{args.dataset_infix}_"

    if args.version != "":
        version = f"v{args.version}_"

    if args.type == 'binary':
        classes = 1
        metric = "binary"
    elif args.type == 'multi':
        classes = 5
        metric = "multiclass"

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 25

    video_transform_train = ApplyTransformToKey(
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

    video_transform_val = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    256,
                ),
                CenterCrop(256),
            ]
        ),
    )

    model_resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, force_reload=True)
    model_resnet.blocks[5].proj = nn.Linear(in_features=2048, out_features=classes, bias=True)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=f'checkpoints_{version}{data_suffix}',
                                          filename=f"ckpt-{version}{data_infix}{data_suffix}" + '-{epoch:02d}-{'
                                                                                                'val_loss:.2f}',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger_name = f"model_{version}{data_infix}{args.type}"
    logger = TensorBoardLogger("model_logger", name=logger_name,
                               version='version_${version}' + f"_{logger_name}_{args.logger_comment}")
    print(f"the logger name is: {logger_name}")

    torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        min_epochs=1,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_progress_bar=True,
        strategy="ddp",
        precision=args.precision,
        logger=logger
    )

    train_set = f"datasets/train-{data_infix}{data_suffix}.csv"
    val_set = f"datasets/val-{data_infix}{data_suffix}.csv"

    model = NeuralNetworkModel(
        num_classes=classes,
        model_type=metric,
        train_dataset_file=train_set,
        val_dataset_file=val_set,
        nn_model=model_resnet,
        augmentation_val=video_transform_val,
        augmentation_train=video_transform_train
    )
    print(f"this train set is gonna be used: {train_set}")
    print(f"this val set is gonna be used: {val_set}")

    if args.mode == 'train':
        trainer.fit(model)
    elif args.mode == 'pred':
        checkpoint_path = checkpoint_callback.best_model_path
        trainer.predict(model, checkpoint_path)
