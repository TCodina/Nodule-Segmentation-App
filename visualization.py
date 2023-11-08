import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def show_full_ct(ct_a, slice_ndx=None, fig_size=(20, 50)):
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

    clim = (-1000.0, 1000)
    cmap = 'gray'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size)

    ax1.imshow(ct_a[ci], clim=clim, cmap=cmap)
    ax1.set_title(f'index {ci}')

    ax2.imshow(ct_a[:, cr], clim=clim, cmap=cmap)
    ax2.invert_yaxis()
    ax2.set_title(f'row {cr}')

    ax3.imshow(ct_a[:, :, cc], clim=clim, cmap=cmap)
    ax3.invert_yaxis()
    ax3.set_title(f'col {cc}')


def show_slices(ct_a, mask, indices, columns=1, fig_size=(20, 40)):

    clim_ct = (-1000.0, 1000)
    cmap_ct = 'gray'
    clim_mask = (0, 1)
    cmap_mask = ListedColormap(['black', 'red'], N=2)

    # distinguish single slice case
    if type(indices) == int or len(indices) == 1:

        plt.figure(figsize=fig_size)
        plt.imshow(ct_a, clim=clim_ct, cmap=cmap_ct)
        plt.imshow(mask, clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red
        return

    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=fig_size)

    for i in range(rows):
        for j in range(columns):
            if j + i*columns < len(indices):
                index = indices[j + i*columns]
                axes[i, j].set_title(f"slice {index}")
                axes[i, j].imshow(ct_a[index], clim=clim_ct, cmap=cmap_ct)
                axes[i, j].imshow(mask[index], clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red
            else:
                break

    fig.tight_layout()


def show_nodule(cand_chunk, slice_ndx, cand_mask=None, slices=None, fig_size=(20, 50)):
    """
    Plot candidate chunk together with its mask and center.

    Args:
        cand_chunk (numpy.array):
        cand_mask (numpy.array):
        slice_ndx (int):
        slices (int): number of slices to plot (default to all indices in chunk)
        fig_size (tuple):
    """

    fig = plt.figure(figsize=fig_size)

    clim_ct = (-1000.0, 1000)
    cmap_ct = 'gray'

    clim_mask = (0, 1)
    cmap_mask = ListedColormap(['black', 'red'], N=2)

    i, r, c = cand_chunk.shape  # IRC coordinates
    if slices is None:
        slices = i
    assert slices % 2 != 0, "Slices must be odd!"

    ci = i//2  # center index (assumes i odd)
    cr = r//2  # center row
    cc = c//2  # center column

    for index in range(slices):
        subplot = fig.add_subplot(1, slices, 1 + index)
        subplot.set_title('slice {}'.format(index - slices//2 + slice_ndx))
        plt.imshow(cand_chunk[index - slices//2 + ci], clim=clim_ct, cmap=cmap_ct)
        if cand_mask is not None:
            plt.imshow(cand_mask[index - slices//2 + ci], clim=clim_mask, cmap=cmap_mask, alpha=0.3)  # overlap mask on top in red
            plt.plot(cr, cc, 'rx')  # mark center of candidate
