"""
contains functions and classes for loading and manipulating the dataset for the segmentation task
in different formats.

functions:
    get_series_on_disk() - list of ct series/identifiers accesible on disk
    get_nodule_dataframe() - gives dataframe with all nodules in disk
    get_ct(series_uid) - initialize an instance of the Ct class and cache it on disk
    get_ct_candidate_chunk(series_uid, center_xyz, width_irc) -
    get_ct_sample_size(series_uid) -

Classes:
   j Ct(series_uid) - create an instance of a ct scan from a series_uid
"""

import pandas as pd
import csv
import functools
import glob
import os
import random

from collections import namedtuple
import SimpleITK as sitk  # parser to go from MetaIO format to NumPy arrays

import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache  # local function for caching
from util.util import XyzTuple, xyz2irc

data_dir = './../data/'  # directory where data files are stored
# data_dir = "/content/drive/MyDrive/LUNA_data_set/"

raw_cache = getCache('./../cache_data')  # get cache form this location


def get_series_on_disk():
    mhd_list = glob.glob(data_dir + 'subset*/*.mhd')
    series_on_disk = [os.path.split(p)[-1][:-4] for p in mhd_list]
    return series_on_disk


@functools.lru_cache(1)  # to cache on disk
def get_nodule_dataframe():
    series_on_disk = get_series_on_disk()
    nodule_df = pd.read_csv(data_dir + '/annotations.csv')
    # restrict annotations to on disk only
    nodule_on_disk_df = nodule_df[nodule_df['seriesuid'].isin(series_on_disk)]
    # sort by diameter and reset index
    return nodule_on_disk_df.sort_values('diameter_mm').reset_index(drop=True)


# def get_nodules_from_series(serie_uid):
#     all_nodules = get_nodule_dataframe()
#     nodules_serie = all_nodules[all_nodules['seriesuid'] == serie_uid]
#     return nodules_serie


# Ct class whose elements are ct scans as numpy arrays,
# together with information about the candidates inside them
# and their locations.
class Ct:
    def __init__(self, series_uid):

        self.series_uid = series_uid

        mhd_path = glob.glob(data_dir + f'subset*/{series_uid}.mhd')[0]

        # black-box method to read from the ct format (MetaIO) to numpy array
        ct_mhd = sitk.ReadImage(mhd_path)  # implicitly consumes the .raw file in addition to the passed-in .mhd file
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)  # 3D array
        # send voxels with values < -1000 (outside the patient) to exactly -1000  and > + 1000 (bones and metal) to 1000
        ct_a.clip(-1000, 1000, ct_a)  # if not clipped there are weird shapes corresponding to the tomographer!
        self.ct_a = ct_a

        # store origin, voxel size and direction matrix (matrix to align between IRC with XYZ coordinate system)
        # as namedTuples from metadata in ct_mhd file. This information is specific of each individual ct scan and will
        # be used for changing between IRC and XYZ coordinates
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        # self.origin_irc = xyz2irc(self.origin_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)  TODO: ERASE THIS

        # get all nodules in given series
        df = get_nodule_dataframe()
        self.nodules_df = df[df.seriesuid == series_uid]  # TODO: should I reset indices here?
        # build mask
        self.positive_mask = self.build_annotation_mask(self.nodules_df)
        # list of indices of the ct scan labeling the slices that have at least one nodule
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())


    # TODO: add information of the diammeter to avoid weird rectangle masks
    def build_annotation_mask(self, nodules_pd, threshold_hu=-700):
        """
        Build an array of the size of the corresponding ct scan but with boolean values for each pixel.
        True for nodule and False for non-nodule.
        This annotation-mask is created by building a bounding box around each
        human-annotated nodule's center and setting to True all pixels that are inside these boxes
        AND also have values greater than a given threshold.
        This annotation-mask is used as the ground truth for segmentation.

        Args:
            nodules_pd:
            threshold_hu (int): Minimal value (~ density) from which we consider a pixel inside bounding box
                                to be part of a nodule.

        Return:
            mask_a (np.array): array of the size of the ct scan but with boolean values. True for nodules.
        """

        bounding_box_a = np.zeros_like(self.ct_a, dtype=bool)  # init with all False pixels

        # build a surrounding box for each nodule inside the entire ct scan
        for nodule_xyz in list(zip(nodules_pd.coordX, nodules_pd.coordY, nodules_pd.coordZ)):
            ci, cr, cc = xyz2irc(nodule_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

            # find limits over index direction
            index_radius = 2
            try:
                while self.ct_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.ct_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            # find limits over row direction
            row_radius = 2
            try:
                while self.ct_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.ct_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            # find limits over colum direction
            col_radius = 2
            try:
                while self.ct_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.ct_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # set only pixels inside the box surrounding the nodule to True, all the rest of the ct array stays False
            bounding_box_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1
                ] = True

        # build the mask for the ENTIRE Ct scan by setting to True only the pixels that are inside a box
        # AND are greater than threshold.
        mask_a = bounding_box_a & (self.ct_a > threshold_hu)

        return mask_a

    def get_candidate_chunk(self, center_xyz, width_irc):
        """
        Builds a chunk of data of sizes specified in width_irc, around a center's candidate, from the corresponding
        Ct scan. This chunk is returned as a portion of the Ct scan and as a portion of the positive_mask.

        Args:
            center_xyz (tuple): center of the candidate in xyz system
            width_irc (tuple): sizes (i,r,c) of the desired output ct chunk

        Return:
            ct_chunk (numpy array): chunk of ct data of desired width centered at center_irc
            pos_chunk (numpy array: bool): same chunk but cut it from positive_mask
            center_irc: center of the chunk data in irc system
        """

        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []  # will contain index slice (start:end) for each of the 3D directions
        # create ct cubic chunk
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.ct_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            # handle possible out-of-bound situations
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.ct_a.shape[axis]:
                end_ndx = self.ct_a.shape[axis]
                start_ndx = int(self.ct_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        # get chunks by slicing over each direction
        ct_chunk = self.ct_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)  # cache on disk 1 ct scan at a time
def get_ct(series_uid):
    """
    Initialize an instance of the Ct class and cache it on disk.
    """
    return Ct(series_uid)


# cache on disk differently, if this cache is commented, the caching does not happen  at all! TODO: explain this
@raw_cache.memoize(typed=True)
def get_ct_candidate_chunk(series_uid, center_xyz, width_irc):
    """
    Initialize get_candidate_chunk and cache it on disk.
    """
    ct = get_ct(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.get_candidate_chunk(center_xyz, width_irc)
    return ct_chunk, pos_chunk, center_irc


# cache the size of each CT scan and its positive inidices,
# so not to load the whole scan every time we need its size only
@raw_cache.memoize(typed=True)
def get_ct_size_and_indices(series_uid):
    """
    Returns number of slices and the entire list of positive indices and cache them in disk.
    """
    ct = get_ct(series_uid)
    return int(ct.ct_a.shape[0]), ct.positive_indexes


# Dataset for validation
class NoduleSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,  # every how many series of full dataset we use them for validation
                 is_val=None,  # validation mode
                 series_uid=None,  # if we want samples only from a specific ct scan
                 context_slices_count=3,  # number of extra slices on each side of the central one (treated as channels)
                 use_full_ct=False,  # whether to use all slices of ct scans or only the ones containing nodules
                 ):

        self.context_slices_count = context_slices_count
        self.use_full_ct = use_full_ct

        # build series_list from a single ct scan or all ct scans in disk
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = get_series_on_disk()

        # if in validation mode, restrict list of series to just the validation subsector
        if is_val:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        # if in training mode, removes validation subsector from list so not to train the model with them
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        # list containing the slices to be used of the ct scans (all if use_full_ct, only positive if not)
        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indices = get_ct_size_and_indices(series_uid)  # cached

            if self.use_full_ct:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indices]

        # restrict to nodules belonging to a series in series_list
        df = get_nodule_dataframe()
        self.nodule_df = df[df['seriesuid'].isin(self.series_list)]

        print("{} {} series, {} slices, {} nodules".format(
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[is_val],
            len(self.sample_list),
            len(self.nodule_df)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_full_slice(series_uid, slice_ndx)  # TODO: why do we need to call another function?

    # TODO: Make this and getitem_trainingCrop below static functions by changing a few things.
    def getitem_full_slice(self, series_uid, slice_ndx):
        """
        Get full slices of a given CT scan as 3D Torch tensors where the first dimension is the number of "channels".
        These channels are the slice indices, being slice_ndx the slice in the center and having context_slices_count
        number of extra slices on each side.

        Args:
            series_uid: the CT scan we want
            slice_ndx: the slice we want of the CT scan.

        Return:
            ct_t (3D torch.Tensor): contain the full slices with slice_ndx as the central channel.
            post_t (3D torch.Tensor): contain a single slice (slice_ndx) of positive mask
        """

        # initialize and build ct_t by piking the relevant slices from the ct scan
        ct = get_ct(series_uid)  # cached
        ct_t = torch.zeros((self.context_slices_count * 2 + 1, 512, 512))
        start_ndx = slice_ndx - self.context_slices_count
        end_ndx = slice_ndx + self.context_slices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.ct_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.ct_a[context_ndx].astype(np.float32))

        # build pos_t by picking the single slice_ndx of the positive mask (no extra context slices!)
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx  # return inputs just for logging info later


# TODO: merge these two dataset classes by writing the getitem functions static and split the very few steps they have
# TODO: different with conditionals (by doing it will be much more easy to understand!)
# Dataset for training (subclassing the validation dataset)
class TrainingNoduleSegmentationDataset(NoduleSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2  # TODO: what is this?

    def __len__(self):
        return len(self.nodule_df)
        #return 100  # TODO: WHY THIS? ORIGINAL SET TO 300000

    def shuffle_samples(self):
        # before shuffling the lists are sorted by diameter
        self.nodule_df = self.nodule_df.sample(frac=1)

    def __getitem__(self, ndx):  # overwrite getitem of validation dataset class
        nodule = self.nodule_df.iloc[ndx % len(self.nodule_df)]
        return self.getitem_training_crop(nodule)

    def getitem_training_crop(self, nodule):  # TODO: make static
        """
        Get pseudo-random 7x64x64 crop of entire ct scan around candidate's center

        Args:
            candidateInfo_tup (NamedTuple):

        Return:
             ct_t (torch.Tensor): 7x64x64 chunk around candidate's center
             pos_t (torch.Tensor): 1x64x64 chunk of mask (boolean values denoting nodule vs non-nodule)
        """
        # chunk of 7 slides of size 96x96 each
        center_xyz = (nodule.coordX, nodule.coordY, nodule.coordZ)
        ct_a, pos_a, center_irc = get_ct_candidate_chunk(nodule.seriesuid, center_xyz, (7, 96, 96))
        pos_a = pos_a[3:4]  # pick center slice of positive mask

        # picks random 64x64 crop inside original 96x96
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, nodule.seriesuid, slice_ndx


# TODO: KG UNDERSTAND THIS
class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1  # candidate_t, pos_t, series_uid, center_t


class TvTrainingLuna2dSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, isValSet_bool=False, val_stride=10, contextSlices_count=3):
        assert contextSlices_count == 3
        data = torch.load('./imgs_and_masks.pt')
        suids = list(set(data['suids']))
        trn_mask_suids = torch.arange(len(suids)) % val_stride < (val_stride - 1)
        trn_suids = {s for i, s in zip(trn_mask_suids, suids) if i}
        trn_mask = torch.tensor([(s in trn_suids) for s in data["suids"]])
        if not isValSet_bool:
            self.imgs = data["imgs"][trn_mask]
            self.masks = data["masks"][trn_mask]
            self.suids = [s for s, i in zip(data["suids"], trn_mask) if i]
        else:
            self.imgs = data["imgs"][~trn_mask]
            self.masks = data["masks"][~trn_mask]
            self.suids = [s for s, i in zip(data["suids"], trn_mask) if not i]
        # discard spurious hotspots and clamp bone
        self.imgs.clamp_(-1000, 1000)
        self.imgs /= 1000

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        oh, ow = torch.randint(0, 32, (2,))
        sl = self.masks.size(1) // 2
        return self.imgs[i, :, oh: oh + 64, ow: ow + 64], 1, self.masks[i, sl: sl + 1, oh: oh + 64, ow: ow + 64].to(
            torch.float32), self.suids[i], 9999
