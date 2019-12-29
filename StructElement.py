# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:24:12 2018

@author: MuhammadSalman
"""

import numpy as np



def square(width, dtype=np.uint8):

    return np.ones((width, width), dtype=dtype)


def rectangle(width, height, dtype=np.uint8):

    return np.ones((width, height), dtype=dtype)


def diamond(radius, dtype=np.uint8):

    L = np.arange(0, radius * 2 + 1)
    I, J = np.meshgrid(L, L)
    return np.array(np.abs(I - radius) + np.abs(J - radius) <= radius,
                    dtype=dtype)


def disk(radius, dtype=np.uint8):

    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)



