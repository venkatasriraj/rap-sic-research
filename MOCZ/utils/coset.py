import numpy as np

def con_coset(n):

    coset = {}
    leader = -1
    visited = set()

    for i in range(n):
        if i in visited:
            continue
        else:
            leader = i
            cos_lis = np.array([])
            while i not in cos_lis:
                cos_lis = np.append(cos_lis, i).astype(int)
                visited.add(i)
                i *= 2
                i %= n
            coset[leader] = cos_lis
    
    return coset