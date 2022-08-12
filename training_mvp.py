import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate  # fancy enumerate()
from data_set import LunaDataset
from util.logconf import logging
from model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


def logMetrics(
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5,
):
    """
    Display full and per-class statistics (loss and correct percentage).
    :param epoch_ndx: just to display the current epoch
    :param mode_str: train or validation
    :param metrics_t:
    :param classificationThreshold:
    :return:
    """

    # array of True (non-nodule) and False (nodule) for labels and predictions.
    # to be used as indices to pick metrics with negative values
    negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
    negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

    # to be used as indices to pick metrics with positive values
    posLabel_mask = ~negLabel_mask  # Bitwise NOT to interchange True and False elementwise
    posPred_mask = ~negPred_mask

    neg_count = int(negLabel_mask.sum())
    pos_count = int(posLabel_mask.sum())

    neg_correct = int((negLabel_mask & negPred_mask).sum())
    pos_correct = int((posLabel_mask & posPred_mask).sum())

    # store metrics in dictionary
    metrics_dict = {'loss/all': metrics_t[METRICS_LOSS_NDX].mean(),
                    'loss/neg': metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean(),  # mask picks only negatives
                    'loss/pos': metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean(),  # picks only positive
                    'correct/all': (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100,
                    'correct/neg': neg_correct / np.float32(neg_count) * 100,
                    'correct/pos': pos_correct / np.float32(pos_count) * 100}

    # display metrics
    # log for all
    log.info(("E{} {:8} {loss/all:.4f} loss, {correct/all:-5.1f}% correct, "
              ).format(epoch_ndx,
                       mode_str,
                       **metrics_dict  # use dictionary keys
                       ))
    # log for negative class
    log.info(("E{} {:8} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
              ).format(epoch_ndx,
                       mode_str + '_neg',
                       neg_correct=neg_correct,
                       neg_count=neg_count,
                       **metrics_dict
                       ))
    # log for positive class
    log.info(("E{} {:8} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
              ).format(epoch_ndx,
                       mode_str + '_pos',
                       pos_correct=pos_correct,
                       pos_count=pos_count,
                       **metrics_dict
                       ))


class LunaTrainingApp:
    def __init__(self, data_dir="data/", epochs=1, batch_size=4, num_workers=0):

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # timestamp to identify training runs

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.num_devices = torch.cuda.device_count()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(self.num_devices))
        self.totalTrainingSamples_count = 0

        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model = LunaModel().to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initDataLoader(self, isValSet_bool=False):
        """
        Initialize dataloader for training or validation set
        :param isValSet_bool: determines whether to use training or validation set
        :return: dataloader
        """
        dataset = LunaDataset(
            val_stride=10,
            isValSet_bool=isValSet_bool,
            data_dir=self.data_dir,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

        return dataloader

    def main(self):
        """
        Here is where the magic happens. It trains the model for all epochs, store the training and
        validation metrics, and display the results.
        """
        log.info("Starting {}...".format(type(self).__name__))

        train_dl = self.initDataLoader(isValSet_bool=False)
        val_dl = self.initDataLoader(isValSet_bool=True)

        for epoch_ndx in range(1, self.epochs + 1):
            log.info("Epoch {} of {}, {} /{} batches (train/val) of size {}".format(
                epoch_ndx,
                self.epochs,
                len(train_dl),
                len(val_dl),
                self.batch_size,
            ))

            # train and returns training metrics for a single epoch
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            # display training metrics
            logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            # return validation metrics for a single epoch
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            # display validation metrics
            logMetrics(epoch_ndx, 'val', valMetrics_t)

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


if __name__ == '__main__':
    LunaTrainingApp().main()
