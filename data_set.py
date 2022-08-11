import copy
import csv
import functools
import glob
import os
from collections import namedtuple
import SimpleITK as sitk  # parser to go from MetaIO format to NumPy arrays
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
from util.disk import getCache
from util.util import XyzTuple, xyz2irc

raw_cache = getCache('part2ch10_raw')

# cleaned and organized way of storing information for each candidate
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',  # name of the tuple
    'isNodule_bool, diameter_mm, series_uid, center_xyz',  # name of the elements
    )


# to cache on disk
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    """
    Return a cleaned, ordered and organized form of the human-annotated files by combining information present in
    candidates and annotations files. If requireOnDisk_bool = True, consider elements of the human-annotated files
    only if they correspond to a ct scan present on disk.

    Output: List of CandidateInfoTuple objects, sorted by diameter in decreasing order.
    """

    # construct a set with only the series_uids present on disk.
    # allows to use data, even if all subsets weren't downloaded
    mhd_list = glob.glob('data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # each element of diameter_dict corresponds to a unique ct scan
    # with a list with the centers and diameters for each nodule in that ct scan
    # 'series_uid': [(center_nod1, diam_nod1), (center_nod2, diam_nod2), ...]
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # TODO: the following lines should run only for series_uid in disk

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []  # List of CandidateInfoTuple objects
    with open('data/candidates.csv', "r") as f:
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
    def __init__(self, series_uid):
        mhd_path = glob.glob('data/subset*/{}.mhd'.format(series_uid))[0]

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
        :param: center_xyz: center of the candidate in xyz system
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
def getCt(series_uid):
    """
    Initialize an instance of the Ct class
    """
    return Ct(series_uid)


# to cache on disk differently TODO: explain this
#@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """
    Initialize an instance of the Ct class and calls getRawCandidate
    """
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        print("{} {} samples".format(len(self.candidateInfo_list), "validation" if isValSet_bool else "training"))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        """
        :param: ndx: index of the candidate
        :return: tuple containing
            (chunk of data surrounding  the candidate, [not is_Nodule, is_Nodule], series_uid, center_irc)
        """
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)  # add channel dimension

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
