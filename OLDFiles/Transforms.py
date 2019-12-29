# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:26:06 2018

@author: MuhammadSalman
"""

import numpy as np


def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x

def distance_transform(bitmap):
    f = np.where(bitmap, 0.0, np.inf)
    for i in range(f.shape[0]):
        _upscan(f[i,:])
        _upscan(f[i,::-1])
    for i in range(f.shape[1]):
        _upscan(f[:,i])
        _upscan(f[::-1,i])
    np.sqrt(f,f)
    return f