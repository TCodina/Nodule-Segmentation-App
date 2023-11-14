import datetime
import os
import torch
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import NoduleSegmentationDataset
from model import UNetWrapper
from transformations import TransformationTrain, TransformationValidation


class TrainingApp:
    def __init__(self, series_trn, series_val,
                 num_workers=2, batch_size=8, epochs=10,
                 validation_freq=None, training_freq=2, save_path=''):

        self.series_trn = series_trn
        self.series_val = series_val

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.epochs = epochs
        # initialize model and optimizer
        self.model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4,
                                 padding=True, batch_norm=True, up_mode='upconv').to(self.device)
        self.optimizer = Adam(self.model.parameters())
        if validation_freq is None:
            self.validation_freq = self.epochs // 10
        else:
            self.validation_freq = validation_freq
        self.training_freq = training_freq
        # keep track of epoch loss
        self.metrics_dict = {'loss_trn': [], 'recall_trn': [], 'precision_trn': [], 'f1_trn': [],
                             'loss_val': [], 'recall_val': [], 'precision_val': [], 'f1_val': []}
        # instantiate transformations
        self.transformation_trn = TransformationTrain()
        self.transformation_val = TransformationValidation()
        self.save_path = save_path

    def main(self):
        """
        Here is where the magic happens. It trains the model for all epochs, store the training and
        validation metrics, and display the results.
        """
        print("Loading data...")
        loader_train, loader_val = self.init_dataloader()

        print(f"\nStart training with {len(loader_train)}/{len(loader_val)} batches (trn/val) "
              f"of size {self.batch_size} for {self.epochs} epochs\n")
        for epoch_ndx in range(1, self.epochs + 1):
            # train and store training loss for a single epoch
            self.train(epoch_ndx, loader_train)
            # validation and logging every validation_freq epochs and at the beginning and end
            if epoch_ndx == 1 or epoch_ndx % self.validation_freq == 0 or epoch_ndx == self.epochs:
                self.validate(epoch_ndx, loader_val)
                self.save_model(epoch_ndx)

    def init_dataloader(self):
        """
        Initialize dataloader for training or validation set

        :param isValSet_bool: determines whether to use training or validation set
        :return: dataloader
        """

        dataset_trn = NoduleSegmentationDataset(self.series_trn, is_val=False, transform=self.transformation_trn)
        dataset_val = NoduleSegmentationDataset(self.series_val, is_val=True, transform=self.transformation_val)
        loader_trn = DataLoader(dataset_trn, shuffle=True,
                                batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.use_cuda)
        loader_val = DataLoader(dataset_val, shuffle=False,
                                batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.use_cuda)

        return loader_trn, loader_val

    def train(self, epoch_ndx, trn_dl):
        """
        Train the model for one epoch by the standard procedure:
         For each batch in trainDl:
            Compute the loss
            Apply backprop to get gradients
            Update model's parameters by performing an optimizer step
        It also stores the training metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param trn_dl:
        :return: training metric
        """
        print(f"Training E{epoch_ndx}")
        self.model.train()
        if epoch_ndx == 1 or epoch_ndx % self.training_freq == 0 or epoch_ndx == self.epochs:
            loss_epoch = 0  # accumulated loss during training
            recall_epoch = 0  # accumulated recall
            precision_epoch = 0  # accumulated precision
            f1_epoch = 0  # accumulated f1 score
        for batch_ndx, batch_tup in enumerate(trn_dl):
            self.optimizer.zero_grad()
            # decomposed batch and send to device
            chunk, mask = batch_tup
            chunk = chunk.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            # output of model
            prediction = self.model(chunk)
            # compute loss
            loss_batch = compute_loss(prediction, mask)
            # compute gradient
            loss_batch.backward()
            # update parameters
            self.optimizer.step()

            with torch.no_grad():
                if epoch_ndx == 1 or epoch_ndx % self.training_freq == 0 or epoch_ndx == self.epochs:
                    # update epoch loss
                    loss_epoch += loss_batch.item()
                    # compute batch metrics
                    recall_batch, precision_batch, f1_batch = compute_metrics(prediction, mask)
                    recall_epoch += recall_batch
                    precision_epoch += precision_batch
                    f1_epoch += f1_batch

        with torch.no_grad():
            if epoch_ndx == 1 or epoch_ndx % self.training_freq == 0 or epoch_ndx == self.epochs:
                # normalize by number of samples
                loss_epoch /= len(trn_dl) * self.batch_size
                recall_epoch /= len(trn_dl) * self.batch_size
                precision_epoch /= len(trn_dl) * self.batch_size
                f1_epoch /= len(trn_dl) * self.batch_size
                # updates metrics dict
                self.metrics_dict['loss_trn'].append(loss_epoch)
                self.metrics_dict['recall_trn'].append(recall_epoch)
                self.metrics_dict['precision_trn'].append(precision_epoch)
                self.metrics_dict['f1_trn'].append(f1_epoch)
                # log metrics
                print(f"Finished: Loss: {round(loss_epoch, 3)}, Recall: {round(recall_epoch, 3)}, "
                      f"Precision: {round(precision_epoch, 3)}, F1: {round(f1_epoch, 3)}\n")

    def validate(self, epoch_ndx, val_dl):
        """
        Compute loss for each batch and return validation metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param val_dl:
        :return: validation metrics
        """
        print(f"Validation E{epoch_ndx}")
        with torch.no_grad():
            self.model.eval()
            loss_epoch = 0
            recall_epoch = 0
            precision_epoch = 0
            f1_epoch = 0
            for batch_ndx, batch_tup in enumerate(val_dl):
                # different from training, here we don't keep the loss for validation
                chunk, mask = batch_tup
                chunk = chunk.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                # output of model
                prediction = self.model(chunk)
                # compute loss
                loss_batch = compute_loss(prediction, mask)
                loss_epoch += loss_batch.item()
                # compute batch metrics
                recall_batch, precision_batch, f1_batch = compute_metrics(prediction, mask)
                recall_epoch += recall_batch
                precision_epoch += precision_batch
                f1_epoch += f1_batch

            # normalize by number of samples
            loss_epoch /= len(val_dl) * self.batch_size
            recall_epoch /= len(val_dl) * self.batch_size
            precision_epoch /= len(val_dl) * self.batch_size
            f1_epoch /= len(val_dl) * self.batch_size
            # updates metrics dict
            self.metrics_dict['loss_val'].append(loss_epoch)
            self.metrics_dict['recall_val'].append(recall_epoch)
            self.metrics_dict['precision_val'].append(precision_epoch)
            self.metrics_dict['f1_val'].append(f1_epoch)
            # log metrics
            print(f"Finished: Loss: {round(loss_epoch, 3)}, Recall: {round(recall_epoch, 3)}, "
                  f"Precision: {round(precision_epoch, 3)}, F1: {round(f1_epoch, 3)}\n")

    def save_model(self, epoch_ndx):
        """
        Save model parameters for the current run.
        If the latter is the best run so far, it creates a second file for it
        """
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # timestamp to identify training runs
        file_path = os.path.join(self.save_path + 'saved_models', f'{time_str}.state')
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
        state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),  # the important part
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),  # to resume training if run interrupted
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'metrics': self.metrics_dict
        }
        torch.save(state, file_path)

        print(f"Saved model params to {file_path}\n")

    def load_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])


def compute_loss(prediction, mask):
    """
    Compute the loss of a given batch using dice loss
    """

    all_loss = dice_loss(prediction, mask)  # for all training samples
    fn_loss = dice_loss(prediction * mask, mask)  # only for pixels included in label (false negatives)

    return all_loss.mean() + fn_loss.mean() * 8  # false negatives with 8X weight


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


def compute_metrics(prediction, mask):
    # true positives, false negatives, and false positives
    tp = (prediction * mask).sum(dim=(1, 2, 3))
    fn = ((1 - prediction) * mask).sum(dim=(1, 2, 3))
    fp = (prediction * (1 - mask)).sum(dim=(1, 2, 3))
    # metrics for batch
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    # handle 0/0 cases (torch label them as nan, so we convert them to 0) and average over batch
    recall = recall.nan_to_num().sum().item()
    precision = precision.nan_to_num().sum().item()
    f1 = f1.nan_to_num().sum().item()
    return recall, precision, f1
