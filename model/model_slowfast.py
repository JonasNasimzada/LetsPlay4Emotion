import argparse
import os

import numpy as np
import pandas as pd
import pytorchvideo
import torch
import torch.nn as nn
from kinetics import Kinetics
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from pytorchvideo.transforms import RandomShortSideScale, ShortSideScale
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix, AUROC
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo


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

        # Define metrics
        self.f1_score = F1Score(task=self.model_type, num_classes=self.num_classes)
        self.accuracy = Accuracy(task=self.model_type, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.model_type, num_classes=self.num_classes)
        self.precision = Precision(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.recall = Recall(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.confusion_matrix = ConfusionMatrix(task=self.model_type, num_classes=self.num_classes, normalize="true")

        # Define loss function based on model type
        if self.model_type == "binary":
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.004]))
        else:
            self.loss = nn.CrossEntropyLoss()

        self.validation_output_list = []

    def forward(self, x):
        return self.conv_layers(x)

    def configure_optimizers(self):
        # Configuring optimizer and learning rate scheduler
        optimizer = SGD(params=self.parameters(), lr=self.lr, weight_decay=1e-5, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        # Create training DataLoader with weighted sampling
        data = pd.read_csv(self.train_dataset_file, delim_whitespace=True, header=None)
        labels = data.iloc[:, 1].tolist()
        label_counts = {0: labels.count(0), 1: labels.count(1)}
        total_samples = len(labels)
        weights = [1.0 / label_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, total_samples, replacement=True)

        train_dataset = Kinetics(
            self.train_dataset_file,
            video_sampler=sampler,
            weights_sampler=weights,
            weights_total_sampler=total_samples,
            video_path_prefix=self.video_path_prefix,
            clip_sampler=make_clip_sampler('random', self.clip_duration),
            transform=self.augmentation_train,
            decode_audio=False
        )

        return DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker - 1,
                          shuffle=False)

    def _common_step(self, batch, batch_idx):
        # Common step for training and validation
        video, input_label = batch['video'], batch['label']
        if self.model_type == "binary":
            input_label = input_label.to(torch.float32).unsqueeze(1)
        else:
            input_label = input_label.to(torch.int64)
        output_network = self(video)
        loss = self.loss(output_network, input_label)
        return loss, output_network, input_label

    def training_step(self, batch, batch_idx):
        # Training step
        loss, output_network, input_label = self._common_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "output_network": output_network, "input_label": input_label}

    def val_dataloader(self):
        # Create validation DataLoader
        val_dataset = pytorchvideo.data.Kinetics(
            self.val_dataset_file,
            clip_sampler=make_clip_sampler('uniform', self.clip_duration),
            transform=self.augmentation_val,
            decode_audio=False,
            video_path_prefix=self.video_path_prefix
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_worker - 1)

    def validation_step(self, batch, batch_idx):
        # Validation step
        loss, output_network, input_label = self._common_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_output_list.append({"loss": loss, "output_network": output_network, "input_label": input_label})
        return {"loss": loss, "output_network": output_network, "input_label": input_label}

    def on_validation_epoch_end(self):
        # Compute metrics at the end of validation epoch
        output_network = torch.cat([x["output_network"] for x in self.validation_output_list])
        input_label = torch.cat([x["input_label"] for x in self.validation_output_list])

        metrics = {
            "val_f1_score": self.f1_score(output_network, input_label),
            "val_accuracy": self.accuracy(output_network, input_label),
            "val_precision": self.precision(output_network, input_label),
            "val_auroc": self.auroc(output_network, input_label),
            "val_recall": self.recall(output_network, input_label),
        }
        self.log_dict(metrics, on_epoch=True, prog_bar=True, sync_dist=True)

        _fig, _ax = self.confusion_matrix.plot()
        _fig.canvas.draw()
        image_np = np.frombuffer(_fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape(_fig.canvas.get_width_height()[::-1] + (3,))
        writer = SummaryWriter()
        for text in _ax.texts:
            text.set_text(round(float(text.get_text()), 3))
        writer.add_image('confusion_matrix v3', image_np, dataformats='HWC')
        self.logger.experiment.add_figure("val_confusion_matrix v3", _fig, self.current_epoch)
        writer.close()

        self.validation_output_list.clear()


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        slowfast_alpha = 4
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        return [slow_pathway, fast_pathway]


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['train', 'val'])
    parser.add_argument("--type", required=True, choices=['binary', 'multi'])
    parser.add_argument("--dataset_infix", default="", help='optional infix for choosing dataset')
    parser.add_argument("--train_dataset", default="")
    parser.add_argument("--val_dataset", default="")
    parser.add_argument("--logger_comment", default="")
    parser.add_argument("--version", default="", type=int)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--video_path_prefix", default="")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # Setup data suffix and model version
    data_suffix = args.type
    data_infix = f"{args.dataset_infix}_" if args.dataset_infix else ""
    version = f"v{args.version}_" if args.version else "v1"
    classes, metric = (1, "binary") if args.type == 'binary' else (5, "multiclass")

    # Transformations for training and validation datasets
    video_transform_train = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(32),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(256),
            RandomHorizontalFlip(p=0.5),
            PackPathway()
        ]),
    )

    video_transform_val = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(32),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCropVideo(256),
            PackPathway()
        ]),
    )

    video_duration = (32 * 2) / 25  # num_frames * sampling_rate / frames_per_second

    # Load pretrained SlowFast model and modify for classification
    model_slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model_slowfast.blocks[6].proj = nn.Linear(in_features=2304, out_features=classes, bias=True)

    # Callbacks and logger setup
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc', dirpath=f'checkpoints_{version}{data_suffix}', mode="max", save_top_k=10,
        filename=f"ckpt-{version}{data_infix}{data_suffix}" + '-{epoch:02d}-{val_auroc:.2f}', save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger_name = f"model_{version}{data_infix}{args.type}"
    logger = TensorBoardLogger("model_logger", name=logger_name,
                               version=f"version_{version}_{logger_name}_{args.logger_comment}")
    print(f"the logger name is: {logger_name}")

    torch.set_float32_matmul_precision('high')

    # Trainer setup
    trainer = Trainer(
        min_epochs=1,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices,
        callbacks=[lr_monitor, checkpoint_callback],
        enable_progress_bar=True,
        strategy="ddp",
        precision=args.precision,
        logger=logger,
    )

    # Model initialization
    model = NeuralNetworkModel(
        num_classes=classes,
        model_type=metric,
        train_dataset_file=args.train_dataset,
        val_dataset_file=args.val_dataset,
        nn_model=model_slowfast,
        augmentation_val=video_transform_val,
        augmentation_train=video_transform_train,
        clip_duration=video_duration,
        batch_size=args.batch_size,
        video_path_prefix=args.video_path_prefix
    )

    print(f"train dataset set: {args.train_dataset}")
    print(f"val dataset set: {args.val_dataset}")

    # Train, validate, or predict based on the mode
    if args.mode == 'train':
        if args.ckpt:
            trainer.fit(model, ckpt_path=args.ckpt)
        else:
            trainer.fit(model)
    elif args.mode == 'val':
        trainer.validate(model, ckpt_path=args.ckpt)
