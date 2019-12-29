# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:50:00 2018

@author: MuhammadSalman
"""

import pandas as pd
active_px = np.argwhere(result2==1)
Conn = pd.DataFrame(columns = ['row','column'] ,data=active_px)
Conn['diffrow'] = np.append(0,np.diff(Conn.row))
Conn['rowlabel']=np.nan
Conn['rowlabel'][Conn['diffrow']>1]=Conn.row

Conn = Conn.sort_values(by=['column'])
Conn['diffcol'] = np.append(0,np.diff(Conn.column))
Conn = Conn.sort_values(by=['row'])


import queue
def find_connectivity(img, i, j):
    dx = [0, 0, 1, 1, 1, -1, -1, -1]
    dy = [1, -1, 0, 1, -1, 0, 1, -1]
    x = []
    y = []
    q = queue.Queue()
    if img[i][j] == 0:
        q.put((i, j))
    while q.empty() == False:
        u, v = q.get()
        x.append(u)
        y.append(v)
        for k in range(8):
            xx = u + dx[k]
            yy = v + dy[k]
            if img[xx][yy] == 0:
                img[xx][yy] = 2
                q.put((xx, yy))
    return x, y