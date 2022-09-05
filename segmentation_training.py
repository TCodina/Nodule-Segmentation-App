import argparse  # to accept command-line arguments
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # to write metrics data in a format that TensorBoard will consume
import torch
import torch.nn as nn
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate  # fancy enumerate() which also estimate remaining computation time
from util.logconf import logging  # display messages in a formatted way
from segmentation_dataset import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from segmentation_model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7  # true positives
METRICS_FN_NDX = 8  # false negatives
METRICS_FP_NDX = 9  # false positives
METRICS_SIZE = 10


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        # if the caller has no arguments, it takes them from command-line and store them in the list sys_argv
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # excludes element 0, the name of the python file

        # initialize parser
        parser = argparse.ArgumentParser()
        # add positional and optional arguments (whether num-workers or num_workers, later it's called by num_workers)
        parser.add_argument('--num_workers',
                            help='Number of worker processes for background data loading',
                            default=0,
                            type=int,
                            )
        parser.add_argument('--batch_size',
                            help='Batch size to use for training',
                            default=2,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment_flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment_offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment_scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment_rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment_noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--data_dir',
                            help='Directory of data',
                            default="data/",
                            type=str,
                            )
        parser.add_argument('--tb_prefix',
                            default='run_seg',
                            help="Data prefix to use for Tensorboard run.",
                            )

        self.args = parser.parse_args(sys_argv)  # store all given arguments
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # timestamp to identify training runs

        # initialize writers to be used later for writing metrics data and tensorboard
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        # store all augmentation parameters into dictionary
        self.augmentation_dict = {}
        if self.args.augmented or self.args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.args.augmented or self.args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.args.augmented or self.args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.args.augmented or self.args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.args.augmented or self.args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_devices = torch.cuda.device_count()
        # extend batch_size if multiple devices present
        self.batch_size = self.args.batch_size
        if self.use_cuda:
            self.batch_size *= self.num_devices

        # initialize model and optimizer
        self.segmentation_model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True,
                                              up_mode='upconv')
        self.augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        self.optimizer = Adam(self.segmentation_model.parameters())

        self.validation_cadence = 5  # epoch frequency of image logging

    def main(self):
        """
        Here is where the magic happens. It trains the model for all epochs, store the training and
        validation metrics, and display the results.
        """

        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dl = self.initDataLoader(isValSet_bool=False)
        val_dl = self.initDataLoader(isValSet_bool=True)

        best_score = 0.0  # to keep track of best score during epochs

        for epoch_ndx in range(1, self.args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches (trn/val) of size {}".format(
                epoch_ndx,
                self.args.epochs,
                len(trn_dl),
                len(val_dl),
                self.batch_size,
            ))

            # train and returns training metrics for a single epoch
            trnMetrics_t = self.doTraining(epoch_ndx, trn_dl)
            # display training metrics and update tb
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # validation and logging every validation_cadence epochs
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # return validation metrics for a single epoch
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)

                # compute score, display validation metrics and update best score
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', trn_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        #  TODO: close writers after training and validating for all epochs
        self.trn_writer.close()
        self.val_writer.close()

    def initDataLoader(self, isValSet_bool=False):
        """
        Initialize dataloader for training or validation set

        :param isValSet_bool: determines whether to use training or validation set
        :return: dataloader
        """
        if isValSet_bool:
            dataset = Luna2dSegmentationDataset(val_stride=10, isValSet_bool=True, contextSlices_count=3,
                                                data_dir=self.args.data_dir)
        else:
            dataset = TrainingLuna2dSegmentationDataset(val_stride=10, isValSet_bool=False, contextSlices_count=3,
                                                     data_dir=self.args.data_dir)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader

    def doTraining(self, epoch_ndx, trn_dl):
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

        self.segmentation_model.train()
        trn_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(trn_dl.dataset), device=self.device)

        batch_iter = enumerateWithEstimate(
            trn_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=trn_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, trn_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        """
        Compute loss for each batch and return validation metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param val_dl:
        :return: validation metrics
        """
        with torch.no_grad():
            self.segmentation_model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                # different from doTraining, here we don't keep the loss for validation
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        """
        Compute the loss of a given batch and store metrics in the form of labels, predictions and loss.
        :param classificationThreshold:
        :param batch_ndx:
        :param batch_tup:
        :param batch_size:
        :param metrics_g:
        :return: loss of given batch
        """

        input_t, label_t, series_list, _slice_ndx_list = batch_tup  # these components come from how dataset was defined

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # if augmentation is required and training mode, apply transformations over input and label (mask)
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)  # output of model

        # calculate dice loss
        diceLoss_g = diceLoss(prediction_g, label_g)  # for all training samples
        fnLoss_g = diceLoss(prediction_g * label_g, label_g)  # only for pixels included in label_g (false negatives)

        # collect metrics
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)
        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)  # covert to hard

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])  # true positives
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])  # false negatives
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])  # false positives

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8  # false negatives with 8X weight

    def initTensorboardWriters(self):
        """
        Create writers for the first time (used inside logMetrics) inside runs/ subdirectories
        """
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '_trn_seg_')
            self.val_writer = SummaryWriter(log_dir=log_dir + '_val_seg_')

    def logImages(self, epoch_ndx, mode_str, dl):
        """
        log images into tensorboard
        """
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]  # pick the same 12 CT scans every time
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid, data_dir=self.args.data_dir)

            # select 6 equidistant slices throughout the CT
            for slice_ndx in range(6):
                # get slice
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5
                ct_t, label_t, series_uid, ct_ndx = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                # send it to device
                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()  # pick central slice only

                # build image with RGB (0,1,2) channels
                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                image_a[:, :, 0] += prediction_a & (1 - label_a)  # false positives as red
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5  # false negatives as orange (1 R + 0.5 G)
                image_a[:, :, 1] += prediction_a & label_a  # true positives as green
                image_a *= 0.5
                image_a.clip(0, 1, image_a)  # normalize and clip

                # save data to tensorboard
                writer = getattr(self, mode_str + '_writer')
                writer.add_image(f'{mode_str}/{series_ndx}_prediction_{slice_ndx}', image_a,
                                 self.totalTrainingSamples_count, dataformats='HWC')

                # save ground truth we are using for training (run only once at the beginning of training process)
                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                    image_a[:, :, 1] += label_a  # slice mask in Green

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(f'{mode_str}/{series_ndx}_label_{slice_ndx}', image_a,
                                     self.totalTrainingSamples_count, dataformats='HWC')

                writer.flush()  # prevents TB from getting confused about which data item belongs where.

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        """
        Compute per-epoch metrics: loss mean, true positives, false negatives, false positives, precision, recall
        and f1 score.

        Returns:
            score (float): factor to determine the "best" training run (we use recall for segmentation)
        """

        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        precision = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict = {'loss/all': metrics_a[METRICS_LOSS_NDX].mean(),
                        'percent_all/tp': sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100,
                        'percent_all/fn': sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100,
                        'percent_all/fp': sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100,
                        'pr/precision': precision,
                        'pr/recall': recall,
                        'pr/f1_score': 2 * (precision * recall) / ((precision + recall) or 1)}

        log.info(("E{} {:8} {loss/all:.4f} loss, {pr/precision:.4f} precision, {pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score").format(epoch_ndx, mode_str, **metrics_dict))
        log.info(("E{} {:8} {loss/all:.4f} loss, {percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, "
                  + "{percent_all/fp:-9.1f}% fp").format(epoch_ndx, mode_str + '_all', **metrics_dict))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar('seg_' + key, value, self.totalTrainingSamples_count)
        writer.flush()

        score = metrics_dict['pr/recall']  # we use recall as the score to quantify the "best" training run

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        """
        Save model parameters for the current run. If the latter is the best run so far, it creates a second file for it
        """

        file_path = os.path.join('saved_models', self.args.tb_prefix,
                                 f'{type_str}_{self.time_str}.{self.totalTrainingSamples_count}.state')

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),  # the important part
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),  # to resume training if run interrupted
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        # create second file if the current run is the best so far
        if isBest:
            best_path = os.path.join('saved_models', self.args.tb_prefix, f'{type_str}_{self.time_str}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("New best model reached, saved params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


def diceLoss(prediction_g, label_g, epsilon=1):
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
    diceLabel_g = label_g.sum(dim=[1, 2, 3])
    dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
    diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])
    # compute Dice coefficient
    diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

    return 1 - diceRatio_g  # to make it a loss


if __name__ == '__main__':
    SegmentationTrainingApp().main()
