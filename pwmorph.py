from StructElement import diamond
import numpy as np
def se2flatidx(f,Bc):

    from numpy import array, where
    h,w=Bc.shape
    h //= 2
    w //= 2
    Bi=[]
    for i,j in zip(*where(Bc)):
        Bi.append( (j-w)+(i-h)*f.shape[1] )
    return array(Bi)


def label(f, Bc=None):

    padsize = 1
    if Bc is None: Bc = diamond(padsize)

    f = np.pad(f,padsize,'constant')
    neighbours = se2flatidx(f,Bc)
    labeled = f * 0
    f = f.ravel()
    labeledflat=labeled.ravel()
    label = 1
    queue = []
    for i in range(f.size):
        if f[i] and labeledflat[i] == 0:
            labeledflat[i]=label
            queue=[i+bi for bi in neighbours]
            while queue:
                ni=queue.pop()
                if f[ni] and labeledflat[ni] == 0:
                    labeledflat[ni]=label
                    for n in neighbours+ni:
                        queue.append(n)
            label += 1
    return labeled[padsize:-padsize,padsize:-padsize] # revert the pad4n() call above