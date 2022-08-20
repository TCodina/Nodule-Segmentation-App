import argparse  # to accept command-line arguments
import datetime
import os
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # to write metrics data in a format that TensorBoard will consume
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate  # fancy enumerate() which also estimate remaining computation time
from util.logconf import logging  # display messages in a formatted way
from data_set import LunaDataset
from model import LunaModel

log = logging.getLogger(__name__)  # Instance of logging for this file
log.setLevel(logging.DEBUG)  # set logging to minimal severity level, so every message is displayed

# Used for computeBatchLoss and logMetrics to indexing metrics (label, predictions and loss)
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
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
        parser.add_argument('--balanced',
                            help="Balance the training data to half positive, half negative.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-noise',
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
                            default='run0',
                            help="Data prefix to use for Tensorboard run",
                            )

        self.args = parser.parse_args(sys_argv)  # store all given arguments
        self.time_str = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H.%M.%S')  # timestamp to identify training runs (TODO: Where do we use this?)

        # initialize writers to be used later for writing metrics data and tensorboard
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.args.augmented or self.args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.args.augmented or self.args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
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
        self.model = LunaModel().to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def main(self):
        """
        Here is where the magic happens. It trains the model for all epochs, store the training and
        validation metrics, and display the results.
        """
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        trn_dl = self.initDataLoader(isValSet_bool=False,
                                     ratio_int=int(self.args.balanced),
                                     augmentation_dict=self.augmentation_dict
                                     )
        val_dl = self.initDataLoader(isValSet_bool=True)

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
            # display training metrics and TODO: something with tb
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # return validation metrics for a single epoch
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            # display validation metrics TODO: something with tb
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        #  TODO: close writers after training and validating for all epochs
        self.trn_writer.close()
        self.val_writer.close()

    def initDataLoader(self,
                       isValSet_bool=False,
                       ratio_int=0,
                       augmentation_dict=None
                       ):
        """
        Initialize dataloader for training or validation set
        :param augmentation_dict:
        :param ratio_int:
        :param isValSet_bool: determines whether to use training or validation set
        :return: dataloader
        """
        dataset = LunaDataset(
            val_stride=10,
            isValSet_bool=isValSet_bool,
            ratio_int=ratio_int,
            augmentation_dict=augmentation_dict,
            data_dir=self.args.data_dir,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader

    def doTraining(self, epoch_ndx, train_dl):
        """
        Train the model for one epoch by the standard procedure:
         For each batch in trainDl:
            Compute the loss
            Apply backprop to get gradients
            Update model's parameters by performing an optimizer step
        It also returns the training metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param train_dl:
        :return: training metric
        """
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        """
        Compute loss for each batch and return validation metrics
        :param epoch_ndx: the current epoch just to display it at some point
        :param val_dl:
        :return: validation metrics
        """
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                # different from doTraining, here we don't keep the loss for validation
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        """
        Compute the loss of a given batch and store metrics in the form of labels, predictions and loss.
        :param batch_ndx:
        :param batch_tup:
        :param batch_size:
        :param metrics_g:
        :return: loss of given batch
        """
        input_t, label_t, _series_list, _center_list = batch_tup  # these components come from how dataset was defined

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)  # output of model

        # calculate loss
        loss_func = nn.CrossEntropyLoss(reduction='none')  # reduction=none to get loss per sample
        loss_g = loss_func(logits_g, label_g[:, 1])

        # calculate metrics
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()  # don't keep track of gradients
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    # create writers for the first time (used inside logMetrics) inside runs/ subdirectories
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.args.tb_prefix, self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn')  # TODO: what is this object?
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val')

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        """
        Display full and per-class statistics (loss and correct percentage).
        :param epoch_ndx: just to display the current epoch
        :param mode_str: train or val    # create writers for the first time (used inside logMetrics)
        :param metrics_t:
        :param classificationThreshold:
        :return:
        """

        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        # array of True (non-nodule) and False (nodule) for labels and predictions.
        # to be used as indices to pick metrics with negative values
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        # to be used as indices to pick metrics with positive values
        posLabel_mask = ~negLabel_mask  # Bitwise NOT to interchange True and False elementwise
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = int((negLabel_mask & negPred_mask).sum())
        truePos_count = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - trueNeg_count
        falseNeg_count = pos_count - truePos_count

        # some metrics
        precision = truePos_count / np.float32(truePos_count + falsePos_count)
        recall = truePos_count / np.float32(truePos_count + falseNeg_count)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # store metric scalars in dictionary (the "/" is to divide into subdirectories in tensorboard later)
        metrics_dict = {'loss/all': metrics_t[METRICS_LOSS_NDX].mean(),
                        'loss/neg': metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean(),  # mask picks only negatives
                        'loss/pos': metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean(),  # picks only positive
                        'correct/all': (truePos_count + trueNeg_count) / np.float32(metrics_t.shape[1]) * 100,
                        'correct/neg': trueNeg_count / np.float32(neg_count) * 100,
                        'correct/pos': truePos_count / np.float32(pos_count) * 100,
                        'pr/precision': precision,
                        'pr/recall': recall,
                        'pr/f1_score': f1_score}

        # display metrics
        # log for all
        log.info(("E{} {:8} {loss/all:.4f} loss, "
                  + "{correct/all:-5.1f}% correct, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(epoch_ndx, mode_str, **metrics_dict))
        # log for negative class
        log.info(("E{} {:8} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
                  ).format(epoch_ndx,
                           mode_str + '_neg',
                           neg_correct=trueNeg_count,
                           neg_count=neg_count,
                           **metrics_dict
                           ))
        # log for positive class
        log.info(("E{} {:8} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
                  ).format(epoch_ndx,
                           mode_str + '_pos',
                           pos_correct=truePos_count,
                           pos_count=pos_count,
                           **metrics_dict
                           ))

        # all what follows is for tensorboard
        self.initTensorboardWriters()  # initialize writers only if they are not initialized yet

        writer = getattr(self, mode_str + '_writer')

        # add scalar metrics
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)  # (name_of_graph, Y-axis, X-axis)

        # add Precision-Recall (pr) curves
        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise


if __name__ == '__main__':
    LunaTrainingApp().main()
