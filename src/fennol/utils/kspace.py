import numpy as np
import math

def get_reciprocal_space_parameters(reciprocal_cells, cutoff, kmax=30, kthr=1.0e-6):
    # find optimal ewald parameters (preprocessing)
    eps = 1.0e-8
    ratio = eps + 1
    x = 0.5
    i = 0
    # approximate value
    while ratio > eps:
        x *= 2
        ratio = math.erfc(x * cutoff) / cutoff
    # refine with binary search
    k = i + 60
    xlo = 0.0
    xhi = x
    for i in range(1, k + 1):
        x = (xlo + xhi) / 2.0
        ratio = math.erfc(x * cutoff) / cutoff
        if ratio > eps:
            xlo = x
        else:
            xhi = x
    bewald = x

    # set k points
    kxs = np.arange(kmax + 1)
    kxs = np.concatenate((kxs, -kxs[1:]))
    k = np.array(np.meshgrid(kxs, kxs, kxs)).reshape(3, -1).T[1:]
    # set exp factor
    m2s = []
    expfacs = []
    ks = []
    nks = []
    for i, A in enumerate(range(reciprocal_cells.shape[0])):
        A = reciprocal_cells[i].T
        m2 = np.sum(
            (
                k[:, 0, None] * A[None, 0, :]
                + k[:, 1, None] * A[None, 1, :]
                + k[:, 2, None] * A[None, 2, :]
            )
            ** 2,
            axis=-1,
        )
        a2 = (np.pi / bewald) ** 2
        expfac = np.exp(-a2 * m2) / m2
        isort = np.argsort(expfac)[::-1]
        expfac = expfac[isort]
        m2 = m2[isort]
        ki = k[isort]
        sel = (expfac > kthr).nonzero()[0]
        nks.append(len(sel))
        m2s.append(m2)
        expfacs.append(expfac)
        ks.append(ki)

    ks = np.stack(ks, axis=0)
    m2s = np.stack(m2s, axis=0)
    expfacs = np.stack(expfacs, axis=0)
    nks = np.array(nks, dtype=np.int64)
    nk = np.max(nks)
    return ks[:, :nk, :], expfacs[:, :nk], m2s[:, :nk], bewald
