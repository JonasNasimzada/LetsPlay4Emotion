import argparse
import os

import pandas as pd
import pytorchvideo
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
    UniformTemporalSubsample, Normalize
)
from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix, AUROC
from torchvision.transforms import Compose, Lambda
from torchvision.transforms import (
    RandomCrop,
    RandomHorizontalFlip
)
from torchvision.transforms._transforms_video import (
    NormalizeVideo, CenterCropVideo
)
from kinetics import Kinetics
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from pretrained_models import Resnet50_FER


class NeuralNetworkModel(LightningModule):
    def __init__(self, num_classes, model_type, train_dataset_file, val_dataset_file, nn_model, augmentation_train,
                 augmentation_val, clip_duration, batch_size, video_path_prefix):
        super().__init__()
        self.video_path_prefix = video_path_prefix
        self.model_type = model_type
        self.train_dataset_file = train_dataset_file
        self.val_dataset_file = val_dataset_file
        self.augmentation_train = augmentation_train
        self.augmentation_val = augmentation_val
        self.conv_layers = nn_model
        self.lr = 0.01
        self.batch_size = batch_size
        self.num_worker = 8
        self.clip_duration = clip_duration
        self.num_classes = num_classes

        self.f1_score = F1Score(task=self.model_type, num_classes=self.num_classes)
        self.accuracy = Accuracy(task=self.model_type, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.model_type, num_classes=self.num_classes)
        self.precision = Precision(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.recall = Recall(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.confusion_matrix = ConfusionMatrix(task=self.model_type, num_classes=self.num_classes, normalize="true")

        if self.model_type == "binary":
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.004]))
        else:
            self.loss = nn.CrossEntropyLoss()

        self.validation_output_list = []

    def forward(self, x):
        x = self.conv_layers(x)
        return x

    def configure_optimizers(self):
        opt = SGD(params=self.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.9)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        data = pd.read_csv(self.train_dataset_file, delimiter='\t', header=None)
        labels = data.iloc[:, 1].tolist()
        label_counts = {0: labels.count(0), 1: labels.count(1)}
        total_samples = len(labels)
        weights = [1.0 / label_counts[label] for label in labels]

        sampler = WeightedRandomSampler(weights, total_samples, replacement=True)

        train_dataset = Kinetics(self.train_dataset_file, video_sampler=sampler, weights_sampler=weights,
                                 weights_total_sampler=total_samples, video_path_prefix=self.video_path_prefix,
                                 clip_sampler=make_clip_sampler('uniform', self.clip_duration),
                                 transform=self.augmentation_train, decode_audio=False)

        loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker,
                            shuffle=False)
        return loader

    def _common_step(self, batch, batch_idx):
        video, input_label = batch['video'], batch['label']
        if self.model_type == "binary":
            input_label = input_label.to(torch.float32).unsqueeze(1)
            output_network = self.forward(video)
            loss = self.loss(output_network, input_label)
        else:
            input_label = input_label.to(torch.int64)
            output_network = self.forward(video)
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
        val_dataset = pytorchvideo.data.Kinetics(self.val_dataset_file,
                                                 clip_sampler=make_clip_sampler('random', self.clip_duration),
                                                 transform=self.augmentation_val, decode_audio=False,
                                                 video_path_prefix=self.video_path_prefix)
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

        self.confusion_matrix.update(output_network, input_label)
        self.f1_score(output_network, input_label)
        self.accuracy(output_network, input_label)
        self.precision(output_network, input_label)
        self.auroc(output_network, input_label)
        self.recall(output_network, input_label)

        self.log_dict(
            {
                "val_f1_score": self.f1_score,
                "val_accuracy": self.accuracy,
                "val_precision": self.precision,
                "val_auroc": self.auroc,
                "val_recall": self.recall,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        _fig, _ax = self.confusion_matrix.plot()

        _fig.canvas.draw()
        image_np = np.frombuffer(_fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape(_fig.canvas.get_width_height()[::-1] + (3,))

        # Log the numpy array as an image to TensorBoard
        writer = SummaryWriter()  # Initialize SummaryWriter
        writer.add_image('confusion_matrix', image_np, dataformats='HWC')  # Add the image to TensorBoard
        self.logger.experiment.add_figure("val_confusion_matrix v2", _fig, self.current_epoch)
        writer.close()

        confusion_matrix_computed = self.confusion_matrix.compute().detach().cpu().numpy().astype(int)
        if self.model_type == "binary":
            df_cm = pd.DataFrame(confusion_matrix_computed, index=range(2), columns=range(2))
        else:
            df_cm = pd.DataFrame(confusion_matrix_computed, index=range(self.num_classes),
                                 columns=range(self.num_classes))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap='Spectral')
        sns.color_palette("magma", as_cmap=True)
        self.logger.experiment.add_figure("val_confusion_matrix", fig, self.current_epoch)

        self.validation_output_list.clear()

    def predict_step(self, batch, batch_idx, dataloader=0):
        video, input_label = batch['video'], batch['label']
        output_network = self.forward(video)
        prediction = torch.argmax(output_network, dim=1)
        return prediction


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['train', 'pred'])
    parser.add_argument("--type", required=True, choices=['binary', 'multi'])
    parser.add_argument("--dataset_infix", default="", help='optional infix for choosing dataset')
    parser.add_argument("--dataset_path", default="datasets")
    parser.add_argument("--logger_comment", default="")
    parser.add_argument("--version", default="", type=int)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--model_ckpt", default="")
    parser.add_argument("--prefix_data", default="")
    parser.add_argument("--video_path_prefix", default="")
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

    slowfast_alpha = 4
    num_clips = 10
    num_crops = 3
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 25

    video_transform_train = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                RandomShortSideScale(
                    min_size=256,
                    max_size=320,
                ),
                RandomCrop(256),
                RandomHorizontalFlip(p=0.5),
                PackPathway()
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
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )
    video_duration = (num_frames * sampling_rate) / frames_per_second

    model_slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model_slowfast.blocks[6].proj = nn.Linear(in_features=2304, out_features=classes, bias=True)

    # model_resnet = Resnet50_FER(args.model_ckpt)
    # model_resnet.model.prediction = nn.Linear(in_features=2048, out_features=2, bias=True)
    checkpoint_dirpath = f'checkpoints_{version}{data_suffix}'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=checkpoint_dirpath,
                                          filename=f"ckpt-{version}{data_infix}{data_suffix}" + '-{epoch:02d}-{val_loss:.2f}',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger_name = f"model_{version}{data_infix}{args.type}"
    logger = TensorBoardLogger("model_logger", name=logger_name,
                               version=f"version_{version}_{logger_name}_{args.logger_comment}")
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

    train_set = f"{args.dataset_path}/train-{data_infix}{data_suffix}.csv"
    val_set = f"{args.dataset_path}/val-{data_infix}{data_suffix}.csv"

    model = NeuralNetworkModel(
        num_classes=classes,
        model_type=metric,
        train_dataset_file=train_set,
        val_dataset_file=val_set,
        nn_model=model_slowfast,
        augmentation_val=video_transform_val,
        augmentation_train=video_transform_train,
        clip_duration=video_duration,
        batch_size=args.batch_size,
        video_path_prefix=args.video_path_prefix
    )
    print(f"this train set is gonna be used: {train_set}")
    print(f"this val set is gonna be used: {val_set}")

    if args.mode == 'train':
        if args.ckpt != "":
            trainer.fit(ckpt_path=args.ckpt, model=model)
        else:
            trainer.fit(model)
    elif args.mode == 'pred':
        checkpoint_path = checkpoint_callback.best_model_path
        trainer.predict(model, checkpoint_path)
