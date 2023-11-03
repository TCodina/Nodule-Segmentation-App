import argparse
import glob
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dataset import Luna2dSegmentationDataset
from dataset_classification import LunaDataset, getCt, getCandidateInfoDict, getCandidateInfoList, CandidateInfoTuple
from model import UNetWrapper
import p2ch14.model
from util.logconf import logging
from util.util import xyz2irc, irc2xyz

from training import TrainingApp

TrainingApp.main()
