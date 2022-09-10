import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def showFullCt(ct_a, slice_ndx=None, fig_size=(20, 50)):
    """
    Plot entire Ct scan as 2D plots seen from the 3 directions passing through the center of the ct scan
    or a specific slice if slice_ndx is given.

    Args:
        ct_a (numpy.array):
        slice_ndx (int): allow to pick particular slice.
        fig_size (tuple):
    """

    ci, cr, cc = [s//2 for s in ct_a.shape]
    if slice_ndx:
        ci = slice_ndx

    fig = plt.figure(figsize=fig_size)

    clim = (-1000.0, 1000)
    cmap = 'gray'

    subplot = fig.add_subplot(1, 3, 1)
    subplot.set_title(f'index {ci}')
    plt.imshow(ct_a[ci], clim=clim, cmap=cmap)

    subplot = fig.add_subplot(1, 3, 2)
    subplot.set_title(f'row {cr}')
    plt.imshow(ct_a[:, cr], clim=clim, cmap=cmap)
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(1, 3, 3)
    subplot.set_title(f'col {cc}')
    plt.imshow(ct_a[:, :, cc], clim=clim, cmap=cmap)
    plt.gca().invert_yaxis()


def showSliceWithMask(ct_slice, ct_slice_mask, slice_ndx, fig_size=(5, 10)):

    clim_ct = (-1000.0, 1000)
    cmap_ct = 'gray'

    clim_mask = (0, 1)
    cmap_mask = ListedColormap(['black', 'red'], N=2)

    plt.figure(figsize=fig_size)
    plt.title(f"slice {slice_ndx}")
    plt.imshow(ct_slice, clim=clim_ct, cmap=cmap_ct)
    plt.imshow(ct_slice_mask, clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red


def showMultipleSlicesWithMask(ct_a, mask, slice_ndx_list, fig_size=(20, 50)):

    clim_ct = (-1000.0, 1000)
    cmap_ct = 'gray'

    clim_mask = (0, 1)
    cmap_mask = ListedColormap(['black', 'red'], N=2)

    fig = plt.figure(figsize=fig_size)
    cols = 3
    rows = int(np.ceil(len(slice_ndx_list) / cols))

    for i, slice_ndx in enumerate(slice_ndx_list):
        subplot = fig.add_subplot(rows, cols, 1 + i)
        plt.title(f"slice {slice_ndx}")
        plt.imshow(ct_a[slice_ndx], clim=clim_ct, cmap=cmap_ct)
        plt.imshow(mask[slice_ndx], clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red


def showCandidate(cand_chunk, slice_ndx, cand_mask=None, fig_size=(20, 50)):
    """
    Plot candidate chunk together with its mask and center.

    Args:
        cand_chunk (numpy.array):
        cand_mask (numpy.array):
        slice_ndx (int):
        fig_size (tuple):
    """

    fig = plt.figure(figsize=fig_size)

    clim_ct = (-1000.0, 1000)
    cmap_ct = 'gray'

    clim_mask = (0, 1)
    cmap_mask = ListedColormap(['black', 'red'], N=2)

    i, r, c = cand_chunk.shape  # IRC coordinates

    index_list = np.round(np.linspace(0, i - 1, min(i, 7))).astype(int).tolist()  # up to 7 images

    cr = r//2  # center row
    cc = c//2  # center column

    for fig_num, index in enumerate(index_list):
        subplot = fig.add_subplot(1, len(index_list), 1 + fig_num)
        subplot.set_title('slice {}'.format(slice_ndx - i//2 + index))
        plt.imshow(cand_chunk[index], clim=clim_ct, cmap=cmap_ct)
        if cand_mask is not None:
            plt.imshow(cand_mask[index], clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red
            plt.plot(cr, cc, 'rx')  # mark center of candidate
