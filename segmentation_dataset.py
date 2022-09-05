import csv
import functools
import glob
import os
import random
from collections import namedtuple

import SimpleITK as sitk

import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('cache_data_raw')  # get cache form this location

# cleaned and organized way of storing information for each candidate  (name tuple, name element1, name  element2, ...)
CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

MaskTuple = namedtuple('MaskTuple',
                       'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, \
                       candidate_mask, lung_mask, neg_mask, pos_mask')


@functools.lru_cache(1)  # to cache on disk
def getCandidateInfoList(requireOnDisk_bool=True, data_dir="data/"):
    """
    TODO
    Return:
        TODO
    """

    # construct a set with only the series_uids present on disk.
    # allows to use data, even if all subsets weren't downloaded
    mhd_list = glob.glob(data_dir + 'subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # TODO: Why do we need to read the two files? Can we just read everything from annotations_with_...?
    candidateInfo_list = []
    with open(data_dir + 'annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,  # isNodule
                    True,  # hasAnnotation
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open(data_dir + 'candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            isNodule_bool = bool(int(row[4]))

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


# TODO: Is this function really necessary?
@functools.lru_cache(1)  # to cache on disk
def getCandidateInfoDict(requireOnDisk_bool=True, data_dir="data/"):
    """
    Just convert the CadidateIntoList into a dictionary where each element has the series_uid as key.
    """
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool, data_dir=data_dir)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict


# Ct class whose elements are ct scans as numpy arrays, together with information about the nodules inside them and
# their locations.
class Ct:
    def __init__(self, series_uid, data_dir="data/"):

        self.data_dir = data_dir
        mhd_path = glob.glob(self.data_dir + 'subset*/{}.mhd'.format(series_uid))[0]

        # black-box method to read from the ct format (MetaIO) to numpy array
        ct_mhd = sitk.ReadImage(mhd_path)  # implicitly consumes the .raw file in addition to the passed-in .mhd file
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)  # 3D array
        # discard voxels with values < -1000 (outside the patient) and > + 1000 (bones and metal)
        ct_a.clip(-1000, 1000, ct_a)
        self.hu_a = ct_a

        self.series_uid = series_uid

        # store origin, voxel size and direction matrix (matrix to align between IRC with XYZ coordinate system)
        # as namedtuples from metadata in ct_mhd file
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict(data_dir=data_dir)[self.series_uid]

        self.positiveInfo_list = [candidate_tup for candidate_tup in candidateInfo_list if candidate_tup.isNodule_bool]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        # list of indices of ct scan, labeling the slices that have at least one nodule
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        """
        Build an array of the size of the corresponding ct scan but with boolean values for each pixel.
        True for nodule and False for non-nodule. This annotation-mask is created by building a bounding box around each
        human-annotated nodule's center and setting to True all pixels that are inside these boxes AND also have values
        greater than a given threshold. This annotation-mask is used as the ground truth for segmentation.

        Args:
            positiveInfo_list: List of positive candidates.
            threshold_hu (int): Minimal value (~ density) from which we consider a pixel inside bounding box
                                to be part of a nodule

        Return:
            mask_a (np.array): array of the size of the ct scan but with boolean values. True for nodules
        """

        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)  # init with all False pixels

        # build a surrounding box for each nodule inside the entire ct scan
        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(candidateInfo_tup.center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            # find limits over index direction
            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            # find limits over row direction
            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            # find limits over colum direction
            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # set only the pixels inside the box surrounding the nodule to True,
            # all the rest of the ct array stays False
            boundingBox_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1
                ] = True

        # build the mask for the ENTIRE Ct scan by setting to True only the pixels that are inside aa box AND
        # are greater than threshold.
        # While boundingBox creates cubes of True pixels, mask reduce them to just the ones forming the nodules.
        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        """
        Builds a chunk of data of sizes specified in width_irc, around a center's candidate,
        from the corresponding Ct scan. This chunk is returned as a portion of the Ct scan and as a portion
        of the positive_mask.

        Args:
            center_xyz: center of the candidate in xyz system
            width_irc: sizes (i,r,c) of the desired output ct chunk

        Return:
            ct_chunk (numpy array): chunk of ct data of desired width centered at center_irc
            pos_chunk (numpy array: bool): same chunk but cut it from positive_mask
            center_irc: center of the chunk data in irc system
        """

        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        # create ct cubic chunk
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            # handle possible out-of-bound situations
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


# cache on disk 1 ct scan at a time
@functools.lru_cache(1, typed=True)
def getCt(series_uid, data_dir="data/"):
    """
    Initialize an instance of the Ct class
    """
    return Ct(series_uid, data_dir=data_dir)


# cache on disk differently, if this cache is commented, the caching does not happen  at all! TODO: explain this
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc, data_dir="data/"):
    """
    Initialize an instance of the Ct class and calls getRawCandidate
    """
    ct = getCt(series_uid, data_dir=data_dir)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, pos_chunk, center_irc


# cache the size of each CT scan and its positive mask, so not to load the whole scan every time we need its size only
@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid, data_dir="data/"):
    """
    Returns number of slices and the entire list of positive indices.
    """
    ct = Ct(series_uid, data_dir=data_dir)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


# Dataset for validation
class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,  # every how many series of full dataset we use them for validation
                 isValSet_bool=None,  # validation mode
                 series_uid=None,  # if we want samples only from a specific ct scan
                 contextSlices_count=3,  # number of extra slices on each side of the central one (treated as channels)
                 fullCt_bool=False,  # whether to use all slices of ct scans or only the ones containing nodules
                 data_dir='data/'  # from where to read the data
                 ):

        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool
        self.data_dir = data_dir

        # build series_list from a single ct scan or all ct scans in disk
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfoDict(data_dir=self.data_dir).keys())

        # if in validation mode, restrict list of series to just the validation subsector
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        # if in training mode, removes validation subsector from list so not to train the model with them
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        # list containing the slices to be used of the ct scans (all if fullCT_bool, only positive if not)
        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid, data_dir=self.data_dir)  # cached

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList(data_dir=self.data_dir)
        series_set = set(self.series_list)  # for faster lookup
        # filter candidates belonging to a series in series_list
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list if cit.series_uid in series_set]
        # positive candidates (in series_list)
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]  # TODO: why this %?
        return self.getitem_fullSlice(series_uid, slice_ndx)  # TODO: why do we need to call another function?

    # TODO: Make this and getitem_trainingCrop below static functions by changing a few things.
    def getitem_fullSlice(self, series_uid, slice_ndx):
        """
        Get full slices of a given CT scan as 3D Torch tensors where the first dimension is the number of "channels".
        These channels are the slice indices, being slice_ndx the slice in the center and having contextSlices_count
        number of extra slices on each side.
        :param series_uid: the CT scan we want
        :param slice_ndx: the slice we want of the CT scan.
        :return:
            ct_t (3D torch.Tensor): contain the full slices with slice_ndx as the central channel.
            post_t (3D torch.Tensor): contain a single slice (slice_ndx) of positive mask
        """

        # initialize and build ct_t by piking the relevant slices from the ct scan
        ct = getCt(series_uid, data_dir=self.data_dir)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))
        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # build pos_t by picking the single slice_ndx of the positive mask
        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx  # return inputs just for logging info later


# Dataset for training (subclassing the validation dataset)
class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        # chunk of 7 slides of size 96x96 each
        ct_a, pos_a, center_irc = getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz,
                                                    (7, 96, 96), data_dir=self.data_dir)
        pos_a = pos_a[3:4]  # pick center slice of postive mask

        # picks random 64x64 crop inside original 96x96
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset + 64, col_offset:col_offset + 64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx


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
