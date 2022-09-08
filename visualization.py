import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#from dataset_segmentation import Ct, LunaDataset
matplotlib.use('nbagg')


def showFullCt(ct_a, slice_ndx=None, fig_size=(20, 50)):
    """
    Plot entire Ct scan as 2D plots seen from the 3 directions passing through the center of the ct scan
    or a specific slice if slice_ndx is given.

    Args:
        ct_a (numpy.array):
        slice_ndx (int): allow to pick particular slice.
        fig_size (tuple):
    """

    origin_irc = [s//2 for s in ct_a.shape]
    if slice_ndx:
        origin_irc[0] = slice_ndx

    fig = plt.figure(figsize=fig_size)

    clim = (-1000.0, 1000)
    cmap = 'gray'

    subplot = fig.add_subplot(1, 3, 1)
    subplot.set_title('index {}'.format(int(origin_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[int(origin_irc[0])], clim=clim, cmap=cmap)

    subplot = fig.add_subplot(1, 3, 2)
    subplot.set_title('row {}'.format(int(origin_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, int(origin_irc[1])], clim=clim, cmap=cmap)
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(1, 3, 3)
    subplot.set_title('col {}'.format(int(origin_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, :, int(origin_irc[2])], clim=clim, cmap=cmap)
    plt.gca().invert_yaxis()


def showFullMask(mask, slice_ndx=None, fig_size=(5, 10)):
    """
    Plot mask of a given Ct scan as a 2D plot passing through the center of the ct scan or a specific slice if
    slice_ndx is given.

    Args:
        mask (numpy.array):
        slice_ndx (int): allow to pick particular slice
        fig_size (tuple):
    """

    if slice_ndx is None:
        slice_ndx = mask.shape[0]//2  # picks center slice

    clim = (0, 1)
    binary_map = ListedColormap(['black', 'red'], N=2)

    plt.figure(figsize=fig_size)
    plt.imshow(mask[slice_ndx], clim=clim, cmap=binary_map)
    plt.title('index {}'.format(slice_ndx), fontsize=30)


# TODO: SEE WHAT TO DO WITH THIS FUNCTION...
def showCandidate(series_uid, batch_ndx=None, **kwargs):
    """
    KG...
    :param series_uid: ct scan series_uid
    :param batch_ndx: index of desired candidate
    :param kwargs:
    :return:
    """
    ds = LunaDataset(series_uid=series_uid, **kwargs)  # get dataset for sample in a single ct scan
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]  # list of positions for nodules

    # define batch_ndx from input or taking first element of pos_list
    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:, int(center_irc[1])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:, :, int(center_irc[2])], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')

    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
