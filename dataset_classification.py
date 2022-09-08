import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple
import SimpleITK as sitk  # parser to go from MetaIO format to NumPy arrays

import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset


log = logging.getLogger(__name__)  # Instance of logging for this file
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)  # set logging to minimal severity level, so every message is displayed

raw_cache = getCache('cache_data_raw')  # get cache form this location

# cleaned and organized way of storing information for each candidate  (name tuple, name element1, name  element2, ...)
CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(1)  # to cache on disk
def getCandidateInfoList(requireOnDisk_bool=True, data_dir="data/"):
    """
    Return a cleaned, ordered and organized form of the human-annotated files by combining information present in
    candidates and annotations files. If requireOnDisk_bool = True, consider elements of the human-annotated files
    only if they correspond to a ct scan present on disk.

    Output: List of CandidateInfoTuple objects, sorted by diameter in decreasing order.
    """

    # construct a set with only the series_uids present on disk.
    # allows to use data, even if all subsets weren't downloaded
    mhd_list = glob.glob(data_dir + 'subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # each element of diameter_dict corresponds to a unique ct scan
    # with a list with the centers and diameters for each nodule in that ct scan
    # 'series_uid': [(center_nod1, diam_nod1), (center_nod2, diam_nod2), ...]
    diameter_dict = {}
    with open(data_dir + 'annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []  # List of CandidateInfoTuple objects
    with open(data_dir + 'candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            # TODO: this for loop runs even if isNodule_bool = 0, but in that case no annotated nodule should coincide

            # check if the current candidate correspond to any of the annotated nodule in the same ct scan
            # by comparing center positions. If so, add diameter information to the candidate, otherwise set it to zero
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                # since the nodule has a non-zero diameter, candidate and nodule centers don't need to coincide exactly
                # but up to some tolerance
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)  # sorting by diameter in decreasing order
    return candidateInfo_list


# Ct class whose elements will be ct scans as numpy arrays
class Ct:
    def __init__(self, series_uid, data_dir="data/"):

        self.data_dir = data_dir
        mhd_path = glob.glob(self.data_dir + 'subset*/{}.mhd'.format(series_uid))[0]

        # black-box method to read from the ct format (MetaIO) to numpy array
        ct_mhd = sitk.ReadImage(mhd_path)  # implicitly consumes the .raw file in addition to the passed-in .mhd file
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)  # 3D array

        # discard voxels with values < -1000 (outside the patient) and > + 1000 (bones and metal)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        # store origin, voxel size and direction matrix (matrix to align between IRC with XYZ coordinate system)
        # as namedtuples from metadata in ct_mhd file
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        """
        :param center_xyz: center of the candidate in xyz system
        :param width_irc: sizes (i,r,c) of the desired output ct chunk
        :return:
            ct_chunk: chunk of ct data of desired width centered at center_irc
            center_irc: center of the chunk data in irc system
        """
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        # create ct cubic chunk
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

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

        return ct_chunk, center_irc


# to cache on disk 1 ct scan at a time
@functools.lru_cache(1, typed=True)
def getCt(series_uid, data_dir="data/"):
    """
    Initialize an instance of the Ct class
    """
    return Ct(series_uid, data_dir=data_dir)


# to cache on disk differently, if this cache is commented, the caching does not happen  at all! TODO: explain this
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc, data_dir="data/"):
    """
    Initialize an instance of the Ct class and calls getRawCandidate
    """
    ct = getCt(series_uid, data_dir=data_dir)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid,
        center_xyz,
        width_irc,
        use_cache=True,
        data_dir="data/"):
    """
    returns Ct candidate but with augmented data. Instead of returning the original Ct chunk, it applies some
    transformations defined on augmentation_dict.
    :param augmentation_dict: dictionary of transformations to be applied on Ct chunk. The allowed transformations are
        'flip', 'offset', 'scale', 'rotate', 'noise'.
    :param series_uid:
    :param center_xyz:
    :param width_irc:
    :param use_cache:
    :param data_dir:
    :return: transformed Ct candidate
    """
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc, data_dir=data_dir)
    else:
        ct = getCt(series_uid, data_dir=data_dir)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    # send chunk to tensor and add channel and batch dimension (5D)
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # initialize 4x4 identity matrix which will be modified by the 5 transformations, one by one, to produce the final
    # complete transformation matrix
    transform_t = torch.eye(4)

    for i in range(3):
        # flip sign randomly of some diagonal components
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        # add translation term (offset) on borders of transformation matrix
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i, 3] = offset_float * random_float

        # scale diagonal elements
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1.0 + scale_float * random_float

    # create rotation matrix over x-y plane of the Ct scan and apply it to the full transformation matrix
    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    # build affine transformation from previous transformation
    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),  # this argument needs to be of size (N, 3, 4) for 3D data
            ct_t.size(),  # target output image size (N,C,D,H,W)
            align_corners=False,
        )
    # get augmented_chunk by applying affine transformation to original chunk
    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    # add noise to the already augmented chunk
    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str="random",
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None,
                 data_dir="data/"):

        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict
        self.data_dir = data_dir

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList(data_dir=self.data_dir))
            self.use_cache = True

        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        # sort the dataset depending on argument sortby_str
        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

        # create list of positive and negative candidates from candidateInfor_list
        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.negative_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.ratio_int:
            return 200000  # fixed length for balanced data
        else:
            return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        """
        :param: ndx: index of the candidate
        :return: tuple containing
            (chunk of data surrounding  the candidate, [not is_Nodule, is_Nodule], series_uid, center_irc)
        """

        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
                data_dir=self.data_dir
            )

        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                data_dir=self.data_dir
            )

            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)  # add channel dimension

        else:
            ct = getCt(candidateInfo_tup.series_uid, data_dir=self.data_dir)
            candidate_a, center_irc = ct.getRawCandidate(candidateInfo_tup.center_xyz, width_irc)
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
