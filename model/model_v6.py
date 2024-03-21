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
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from kinetics import Kinetics
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from video_dataset import VideoFrameDataset, ImglistToTensor
from pretrained_models import Resnet50_FER, Resnet50_FER_V2


class NeuralNetworkModel(LightningModule):
    def __init__(self, num_classes, model_type, train_dataset_file, val_dataset_file, nn_model, augmentation_train,
                 augmentation_val, clip_duration, batch_size, video_path_prefix, annotation_file_train,
                 annotation_file_val):
        super().__init__()
        self.video_path_prefix = video_path_prefix
        self.annotation_file_train = annotation_file_train
        self.annotation_file_val = annotation_file_val
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
        preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = VideoFrameDataset(
            root_path=self.video_path_prefix,
            annotationfile_path=self.annotation_file_train,
            num_segments=5,
            frames_per_segment=1,
            imagefile_template='frame_{:04d}.jpg',
            transform=preprocess,
            test_mode=False
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            pin_memory=True
        )

        return loader

    def _common_step(self, batch, batch_idx):
        video, input_label = batch
        batch_size, num_frames, channels, height, width = video.shape
        video = video.reshape(batch_size * num_frames, channels, height, width)
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
        preprocess = transforms.Compose([
            ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        dataset = VideoFrameDataset(
            root_path=self.video_path_prefix,
            annotationfile_path=self.annotation_file_val,
            num_segments=5,
            frames_per_segment=1,
            imagefile_template='frame_{:04d}.jpg',
            transform=preprocess,
            test_mode=False
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            pin_memory=True
        )
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
    parser.add_argument("--ann_val", default="")
    parser.add_argument("--ann_train", default="")
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
                UniformTemporalSubsample(8),
                Lambda(lambda x: x / 255.0),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(244),
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
                Normalize(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
            ]
        ),
    )
    video_duration = (num_frames * sampling_rate) / frames_per_second

    model_resnet = Resnet50_FER_V2(args.model_ckpt)

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
        nn_model=model_resnet,
        augmentation_val=video_transform_val,
        augmentation_train=video_transform_train,
        clip_duration=video_duration,
        batch_size=args.batch_size,
        video_path_prefix=args.video_path_prefix,
        annotation_file_val=args.ann_val,
        annotation_file_train=args.ann_train
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
