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
    Ct(series_uid) - create an instance of a ct scan from a series_uid
"""

import pandas as pd
import glob
import os
import SimpleITK as sitk  # parser to go from MetaIO format to NumPy arrays
import numpy as np
from torch.utils.data import Dataset

from util.disk import get_cache  # local function for caching
from util.util import XyzTuple, xyz2irc

data_dir = './../data/'  # directory where data files are stored
# data_dir = "/content/drive/MyDrive/LUNA_data_set/"

#cache_data = get_cache('./../cache_data')  # get cache form this location


def get_series_on_disk():
    mhd_list = glob.glob(data_dir + 'subset*/*.mhd')
    series_on_disk = [os.path.split(p)[-1][:-4] for p in mhd_list]
    return series_on_disk


#@functools.lru_cache(1)  # to cache on disk
def get_nodule_dataframe():
    series_on_disk = get_series_on_disk()
    nodule_df = pd.read_csv(data_dir + '/annotations.csv')
    # restrict annotations to on disk only
    nodule_on_disk_df = nodule_df[nodule_df['seriesuid'].isin(series_on_disk)]
    # sort by diameter and reset index
    return nodule_on_disk_df.sort_values('diameter_mm').reset_index(drop=True)


# Ct class whose elements are ct scans as numpy arrays,
# together with information about the nodules inside them and their locations.
class Ct:
    def __init__(self, series_uid):

        self.series_uid = series_uid
        mhd_path = glob.glob(data_dir + f'subset*/{self.series_uid}.mhd')[0]

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
        # build mask of the same size as original ct but with boolean values: 1 for nodule 0 for non-nodule
        self.mask = self.build_mask(self.nodules_df)
        # list of indices of the ct scan labeling the slices that have at least one nodule
        self.positive_indices = (self.mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    # TODO: add information of the diammeter to avoid weird rectangle masks
    def build_mask(self, nodules_df, threshold_hu=-700):
        """
        Build an array of the size of the corresponding ct scan but with boolean values for each pixel.
        True for nodule and False for non-nodule.
        This mask is created by building a bounding box around each
        human-annotated nodule's center and setting to True all pixels that are inside these boxes
        AND also have values greater than a given threshold.
        This annotation-mask is used as the ground truth for segmentation.

        Args:
            nodules_df:
            threshold_hu (int): Minimal value (~ density) from which we consider a pixel inside bounding box
                                to be part of a nodule.

        Return:
            mask_a (np.array): array of the size of the ct scan but with boolean values. True for nodules.
        """

        bounding_box = np.zeros_like(self.ct_a, dtype=bool)  # init with all False pixels

        # build a surrounding box for each nodule inside the entire ct scan
        for nodule_xyz in list(zip(nodules_df.coordX,
                                   nodules_df.coordY,
                                   nodules_df.coordZ)):
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
            bounding_box[
            ci - index_radius: ci + index_radius + 1,
            cr - row_radius: cr + row_radius + 1,
            cc - col_radius: cc + col_radius + 1
            ] = True

        # build the mask for the ENTIRE Ct scan by setting to True only the pixels that are inside a box
        # AND are greater than threshold.
        mask = bounding_box & (self.ct_a > threshold_hu)

        return mask

    def get_cuboid(self, center_xyz, width_irc):
        """
        Builds a 3D chunk of data (cuboid) of sizes specified in width_irc, around the point center_xyz,
        from the corresponding Ct scan.
        The cuboid is returned as a portion of the Ct scan and as a portion of the positive_mask.

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
        mask_chunk = self.mask[tuple(slice_list)]

        return ct_chunk, mask_chunk, center_irc


#@functools.lru_cache(1, typed=True)  # cache on disk 1 ct scan at a time
def get_ct(series_uid):
    """
    Initialize an instance of the Ct class and cache it on disk.
    """
    return Ct(series_uid)


# cache on disk differently, if this cache is commented, the caching does not happen  at all! TODO: explain this
#@cache_data.memoize(typed=True)
def get_ct_cuboid(series_uid, center_xyz, width_irc):
    """
    Initialize get_cuboid and cache it on disk.
    """
    ct = get_ct(series_uid)
    ct_chunk, mask_chunk, center_irc = ct.get_cuboid(center_xyz, width_irc)
    return ct_chunk, mask_chunk, center_irc


# cache the size of each CT scan and its positive inidices,
# so not to load the whole scan every time we need its size only
#@cache_data.memoize(typed=True)
def get_ct_size_and_indices(series_uid):
    """
    Returns number of slices and the entire list of positive indices and cache them in disk.
    """
    ct = get_ct(series_uid)
    return int(ct.ct_a.shape[0]), ct.positive_indices


# Dataset
# TODO: maybe try first using the same getitem for training and testing, what would be the problem with that?
class NoduleSegmentationDataset(Dataset):
    def __init__(self,
                 series_list=None,  # series from which to make the dataset
                 is_val=False,  # validation mode
                 transform=None,
                 context_slices_count=3,  # number of extra slices on each side of the central one (treated as channels)
                 use_full_ct=False,  # whether to use all slices of ct scans or only the ones containing nodules
                 ):

        self.is_val = is_val
        self.context_slices_count = context_slices_count
        self.use_full_ct = use_full_ct
        self.series_list = series_list
        self.transform = transform

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
        if self.is_val:  # length for validation
            return len(self.sample_list)
        else:  # length for training
            return len(self.nodule_df)  # more than actual length because of augmentation

    def __getitem__(self, ndx):
        if self.is_val:
            # Get full slices of a given CT scan as 3D Torch tensors where the first dimension
            # is the number of "channels". These channels are the slice indices, being slice_ndx the slice in the center
            # and having context_slices_count number of extra slices on each side.
            series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
            ct = get_ct(series_uid)  # cached
            # start and end indices taking care of boundaries
            start_ndx = max(0, slice_ndx - self.context_slices_count)
            end_ndx = min(ct.ct_a.shape[0], slice_ndx + self.context_slices_count + 1)
            ct_slices = ct.ct_a[start_ndx:end_ndx]
            # build mask_t by picking the single slice_ndx of the positive mask (no extra context slices!)
            mask_slice = ct.mask[slice_ndx: slice_ndx + 1]
            # transform
            if self.transform:
                ct_slices, mask_slice = self.transform(ct_slices, mask_slice)

            return ct_slices, mask_slice

        else:
            # Get pseudo-random 7x64x64 crop of entire ct scan around candidate's center
            nodule = self.nodule_df.iloc[ndx % len(self.nodule_df)]

            center_xyz = (nodule.coordX, nodule.coordY, nodule.coordZ)
            ct_cub, mask_cub, center_irc = get_ct_cuboid(nodule.seriesuid, center_xyz, (7, 96, 96))
            mask_slice = mask_cub[3:4]  # pick center slice of positive mask
            if self.transform:  # transform
                ct_cub, mask_slice = self.transform(ct_cub, mask_slice)

            return ct_cub, mask_slice
