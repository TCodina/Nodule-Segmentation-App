import datetime
import os

import numpy as np  
import torch
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import NoduleSegmentationDataset, TrainingNoduleSegmentationDataset
from model import UNetWrapper, SegmentationAugmentation


class TrainingApp:
    def __init__(self, num_workers=2, batch_size=8, epochs=10, augmentation_dict=None):

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epochs = epochs
        if augmentation_dict is None:
            self.augmentation_dict = {'flip': True,
                                      'offset': 0.03,
                                      'scale': 0.2,
                                      'rotate': True,
                                      'noise': 25.0}
        else:
            self.augmentation_dict = augmentation_dict
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # timestamp to identify training runs
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # initialize model and optimizer
        self.model = UNetWrapper(in_channels=7,
                                 n_classes=1,
                                 depth=3,
                                 wf=4,
                                 padding=True,
                                 batch_norm=True,
                                 up_mode='upconv').to(self.device)
        self.augment = SegmentationAugmentation(**self.augmentation_dict).to(self.device)
        self.optimizer = Adam(self.model.parameters())
        self.validation_cadence = 2  # epoch frequency of test again validation set

    def main(self):
        """
        Here is where the magic happens. It trains the model for all epochs, store the training and
        validation metrics, and display the results.
        """

        print("Starting training")

        trn_dl = self.init_dataloader(is_val=False)
        val_dl = self.init_dataloader(is_val=True)

        best_score = 0.0  # to keep track of best score during epochs

        for epoch_ndx in range(1, self.epochs + 1):
            print(f"Epoch {epoch_ndx} of {self.epochs}, {len(trn_dl)}/{len(val_dl)} batches (trn/val) /"
                  f"of size {self.batch_size}")

            self.train(epoch_ndx, trn_dl)  # train and returns training metrics for a single epoch

            # validation and logging every validation_cadence epochs
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                self.validate(epoch_ndx, val_dl)
                self.save_model(epoch_ndx)

    def init_dataloader(self, is_val):
        """
        Initialize dataloader for training or validation set

        :param isValSet_bool: determines whether to use training or validation set
        :return: dataloader
        """
        if is_val:
            dataset = NoduleSegmentationDataset(val_stride=10, is_val=True, context_slices_count=3)
        else:
            dataset = TrainingNoduleSegmentationDataset(val_stride=10, is_val=False, context_slices_count=3)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader

    def train(self, epoch_ndx, trn_dl):
        """
        Train the model for one epoch by the standard procedure:
         For each batch in trainDl:
            Compute the loss
            Apply backprop to get gradients
            Update model's parameters by performing an optimizer step
        It also returns the training metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param trn_dl:
        :return: training metric
        """

        self.model.train()
        trn_dl.dataset.shuffle_samples()

        print(f"Starting training at E{epoch_ndx} ----/{len(trn_dl)}")
        for batch_ndx, batch_tup in enumerate(trn_dl):

            if (batch_ndx % (len(trn_dl)//5)) == 0 and batch_ndx != 0:
                print(f"E{epoch_ndx} {batch_ndx:-4}/{len(trn_dl)}")

            self.optimizer.zero_grad()

            loss_var = self.compute_loss(batch_tup)
            loss_var.backward()

            self.optimizer.step()

        print(f"Training E{epoch_ndx} finished.")

    def validate(self, epoch_ndx, val_dl):
        """
        Compute loss for each batch and return validation metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param val_dl:
        :return: validation metrics
        """
        with torch.no_grad():
            self.model.eval()

            print(f"Starting validation at epoch E{epoch_ndx} ----/{len(val_dl)}")
            for batch_ndx, batch_tup in enumerate(val_dl):
                # different from doTraining, here we don't keep the loss for validation

                if (batch_ndx % (len(val_dl)//5)) == 0 and batch_ndx !=0:
                    print(f"E{epoch_ndx} {batch_ndx:-4}/{len(val_dl)}")

                self.compute_loss(batch_tup)

        print(f"Validation E{epoch_ndx} finished.")

    def compute_loss(self, batch_tup):
        """
        Compute the loss of a given batch
        :param batch_tup:
        :return: loss of given batch
        """

        # decomposed batch and send to device
        chunk, mask, series_list, slice_ndx_list = batch_tup
        chunk = chunk.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        # if training mode, apply transformations over input and label (mask)
        if self.model.training:
            chunk, mask = self.augment(chunk, mask)
        # output of model
        prediction = self.model(chunk)
        # calculate dice loss
        all_loss = dice_loss(prediction, mask)  # for all training samples
        fn_loss = dice_loss(prediction * mask, mask)  # only for pixels included in label (false negatives)

        return all_loss.mean() + fn_loss.mean() * 8  # false negatives with 8X weight

    def save_model(self, epoch_ndx):
        """
        Save model parameters for the current run.
        If the latter is the best run so far, it creates a second file for it
        """
        file_path = os.path.join('saved_models', f'{self.time_str}.state')
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),  # the important part
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),  # to resume training if run interrupted
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
        }
        torch.save(state, file_path)

        print(f"Saved model params to {file_path}")


def dice_loss(prediction, label, epsilon=1):
    """
    Compute (soft) Dice loss between predictions and labels (mask)

    Args:
        prediction_g (torch.Tensor 4D): batch of boxes
        label_g (torch.Tensor 4D): batch of masks
        epsilon: regulator to avoid 0/0
    Return:
        1 - Dice coefficient: to make it a loss
    """

    # sum over everything but the batch dimension to get positive labeled, positive predicted, positive correct
    dice_label = label.sum(dim=[1, 2, 3])
    dice_prediction = prediction.sum(dim=[1, 2, 3])
    dice_correct = (prediction * label).sum(dim=[1, 2, 3])
    # compute Dice coefficient
    dice_ratio = (2 * dice_correct + epsilon) / (dice_prediction + dice_label + epsilon)

    return 1 - dice_ratio  # to make it a loss
