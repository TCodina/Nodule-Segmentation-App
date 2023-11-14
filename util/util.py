import collections
import datetime
import time
import numpy as np


IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    """
    Transform CT scan data from IRC to XYZ coordinate system
    :param coord_irc:
    :param origin_xyz:
    :param vxSize_xyz:
    :param direction_a:
    :return:  XyzTuple
    """
    cri_a = np.array(coord_irc)[::-1]  # flip from IRC to CRI
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a  # scale, rotate and add offset for origin
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    """
    Transform CT scan data from XYZ to IRC coordinate system
    :param coord_xyz:
    :param origin_xyz:
    :param vxSize_xyz:
    :param direction_a:
    :return: IrcTuple
    """
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a  # subtract offset, rotate, scale
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))  # flip from RCI to ICR
