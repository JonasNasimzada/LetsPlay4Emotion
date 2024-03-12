import argparse
import os
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter.combinatorics import SamplerIterDataPipe
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix, AUROC
from torchvision import transforms
from torchvision.io import read_video

from pretrained_models import Resnet50_FER


class CustomVideoDataset(IterDataPipe, ABC):
    def __init__(self, dataframe, video_path_prefix, clip_duration, augmentation_train=None):
        self.dataframe = dataframe
        self.video_path_prefix = video_path_prefix
        self.clip_duration = clip_duration
        self.augmentation_train = augmentation_train

        data = pd.read_csv(self.dataframe, delimiter='\t', header=None)
        labels = data.iloc[:, 1].tolist()
        label_counts = {0: labels.count(0), 1: labels.count(1)}
        total_samples = len(labels)
        weights = [1.0 / label_counts[label] for label in labels]

        self.sampler = WeightedRandomSampler(weights, total_samples, replacement=True)

    def __iter__(self):
        for idx in range(len(self.dataframe)):
            video_file, label = self.dataframe.iloc[idx]
            video_path = os.path.join(self.video_path_prefix, video_file)
            frames, _, _ = read_video(video_path)
            frames = frames.permute(0, 3, 1, 2)  # Change shape to (T, C, H, W)

            if self.augmentation_train:
                # Apply augmentation if provided
                frames = self.augmentation_train(frames)

            yield frames, label


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, video_path_prefix, clip_duration, batch_size):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.video_path_prefix = video_path_prefix
        self.clip_duration = clip_duration
        self.batch_size = batch_size
        self.num_workers = 8

        self.augmentation_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random resized crop
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

        # Define augmentation for validation/testing (just normalization)
        self.augmentation_val = transforms.Compose([
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

    def setup(self, stage=None):
        self.train_dataset = CustomVideoDataset(
            self.train_csv,
            video_path_prefix=self.video_path_prefix,
            clip_duration=self.clip_duration,
            augmentation_train=self.augmentation_train  # Provide your train augmentation function
        )
        self.val_dataset = CustomVideoDataset(
            self.val_csv,
            video_path_prefix=self.video_path_prefix,
            clip_duration=self.clip_duration,
            augmentation_train=self.augmentation_val
        )

    def train_dataloader(self):
        data_pipe = SamplerIterDataPipe(self.train_dataset, sampler=self.train_dataset.sampler)
        return DataLoader(data_pipe, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


class NeuralNetworkModel(pl.LightningModule):
    def __init__(self, num_classes, model_type, nn_model):
        super().__init__()
        self.model_type = model_type
        self.conv_layers = nn_model
        self.lr = 0.01
        self.num_worker = 8
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

    def _common_step(self, batch, batch_idx):
        video, input_label = batch
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

    model_resnet = Resnet50_FER(args.model_ckpt)
    model_resnet.model.prediction = nn.Linear(in_features=2048, out_features=2, bias=True)
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

    num_frames = 8
    sampling_rate = 8
    frames_per_second = 25
    video_duration = (num_frames * sampling_rate) / frames_per_second

    train_set = f"{args.dataset_path}/train-{data_infix}{data_suffix}.csv"
    val_set = f"{args.dataset_path}/val-{data_infix}{data_suffix}.csv"
    data_module = VideoDataModule(train_set, val_set, args.video_path_prefix, video_duration, args.batch_size)

    model = NeuralNetworkModel(
        num_classes=classes,
        model_type=metric,
        nn_model=model_resnet,
    )
    print(f"this train set is gonna be used: {train_set}")
    print(f"this val set is gonna be used: {val_set}")

    if args.mode == 'train':
        if args.ckpt != "":
            trainer.fit(ckpt_path=args.ckpt, model=model, datamodule=data_module)
        else:
            trainer.fit(model, datamodule=data_module)
    elif args.mode == 'pred':
        checkpoint_path = checkpoint_callback.best_model_path
        trainer.predict(model, checkpoint_path)
