import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable
import numpy as np
from .atomic_units import AtomicUnits as au
from .xenonpy_props import XENONPY_PROPS

PERIODIC_TABLE_STR = """
H                                                                                                                           He
Li  Be                                                                                                  B   C   N   O   F   Ne
Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
"""

PERIODIC_TABLE = ["Dummy"] + PERIODIC_TABLE_STR.strip().split()

PERIODIC_TABLE_REV_IDX = {s: i for i, s in enumerate(PERIODIC_TABLE)}


def _build_periodic_coordinates():
    periods = PERIODIC_TABLE_STR.split("\n")[1:-1]
    coords = [
        [0, 0],
    ]
    for i, p in enumerate(periods):
        for j in range(0, len(p), 4):
            if p[j : j + 4].strip():
                coords.append([i + 1, j // 4 + 1])
    return coords


PERIODIC_COORDINATES = _build_periodic_coordinates()

ATOMIC_MASSES = [
    0.0,
    1.008,
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,
    14.007,
    15.999,
    18.99840316,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973762,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    208.98243,
    209.98715,
    222.01758,
    223.01974,
    226.02541,
    227.02775,
    232.0377,
    231.03588,
    238.02891,
    237.04817,
    244.06421,
    243.06138,
    247.07035,
    247.07031,
    251.07959,
    252.083,
    257.09511,
    258.09843,
    259.101,
    262.11,
    267.122,
    268.126,
    271.134,
    270.133,
    269.1338,
    278.156,
    281.165,
    281.166,
    285.177,
    286.182,
    289.19,
    289.194,
    293.204,
    293.208,
    294.214,
]


class SpeciesConverter(nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    (!! ROUTINE FROM TORCHANI PACKAGE !!)
    Copyright 2018- Xiang Gao and other ANI developers

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """

    species_order: Sequence[str]

    def setup(self):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())

        conv_tensor = [-1] * (maxidx + 2)
        for i, s in enumerate(self.species_order):
            conv_tensor[rev_idx[s]] = i

        self.conv_tensor = self.variable(
            "buffers", "conv_tensor", jnp.array(conv_tensor, dtype=jnp.int32)
        )

    @nn.compact
    def __call__(self, species):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f"Unknown species found in {species}")

        return converted_species.to(species.device)

    def __repr__(self):
        return f"SpeciesConverter({self.species})"


ELECTRONIC_STRUCTURE = [[0] * 15] * len(PERIODIC_TABLE)
######################### 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p  6s 4f 5d 6p
ELECTRONIC_STRUCTURE[1] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # H
ELECTRONIC_STRUCTURE[2] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # He
ELECTRONIC_STRUCTURE[3] = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Li
ELECTRONIC_STRUCTURE[4] = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Be
ELECTRONIC_STRUCTURE[5] = [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # B
ELECTRONIC_STRUCTURE[6] = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # C
ELECTRONIC_STRUCTURE[7] = [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # N
ELECTRONIC_STRUCTURE[8] = [2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # O
ELECTRONIC_STRUCTURE[9] = [2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # F
ELECTRONIC_STRUCTURE[10] = [2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Ne
########################## 1s 2s 2p 3s 3p 4s 3d 4p  5s 4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[11] = [2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Na
ELECTRONIC_STRUCTURE[12] = [2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Mg
ELECTRONIC_STRUCTURE[13] = [2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Al
ELECTRONIC_STRUCTURE[14] = [2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Si
ELECTRONIC_STRUCTURE[15] = [2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # P
ELECTRONIC_STRUCTURE[16] = [2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # S
ELECTRONIC_STRUCTURE[17] = [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Cl
ELECTRONIC_STRUCTURE[18] = [2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Ar
########################## 1s 2s 2p 3s 3p 4s 3d 4p  5s 4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[19] = [2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # K
ELECTRONIC_STRUCTURE[20] = [2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Ca
ELECTRONIC_STRUCTURE[21] = [2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # Sc
ELECTRONIC_STRUCTURE[22] = [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]  # Ti
ELECTRONIC_STRUCTURE[23] = [2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]  # V
ELECTRONIC_STRUCTURE[24] = [2, 2, 6, 2, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0]  # Cr
ELECTRONIC_STRUCTURE[25] = [2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0]  # Mn
ELECTRONIC_STRUCTURE[26] = [2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0]  # Fe
ELECTRONIC_STRUCTURE[27] = [2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0]  # Co
ELECTRONIC_STRUCTURE[28] = [2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0]  # Ni
ELECTRONIC_STRUCTURE[29] = [2, 2, 6, 2, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Cu
ELECTRONIC_STRUCTURE[30] = [2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Zn
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[31] = [2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0]  # Ga
ELECTRONIC_STRUCTURE[32] = [2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0]  # Ge
ELECTRONIC_STRUCTURE[33] = [2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0]  # As
ELECTRONIC_STRUCTURE[34] = [2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0]  # Se
ELECTRONIC_STRUCTURE[35] = [2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0]  # Br
ELECTRONIC_STRUCTURE[36] = [2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0]  # Kr
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[37] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0]  # Rb
ELECTRONIC_STRUCTURE[38] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0]  # Sr
ELECTRONIC_STRUCTURE[39] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0]  # Y
ELECTRONIC_STRUCTURE[40] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0]  # Zr
ELECTRONIC_STRUCTURE[41] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 4, 0, 0, 0, 0, 0]  # Nb
ELECTRONIC_STRUCTURE[42] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 5, 0, 0, 0, 0, 0]  # Mo
ELECTRONIC_STRUCTURE[43] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0]  # Tc
ELECTRONIC_STRUCTURE[44] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 7, 0, 0, 0, 0, 0]  # Ru
ELECTRONIC_STRUCTURE[45] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 8, 0, 0, 0, 0, 0]  # Rh
ELECTRONIC_STRUCTURE[46] = [2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0, 0, 0, 0]  # Pd
ELECTRONIC_STRUCTURE[47] = [2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0, 0, 0, 0]  # Ag
ELECTRONIC_STRUCTURE[48] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0]  # Cd
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s  4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[49] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0]  # In
ELECTRONIC_STRUCTURE[50] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0]  # Sn
ELECTRONIC_STRUCTURE[51] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0]  # Sb
ELECTRONIC_STRUCTURE[52] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0]  # Te
ELECTRONIC_STRUCTURE[53] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0]  # I
ELECTRONIC_STRUCTURE[54] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0]  # Xe
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s  4d 5p 6s 4f 5d 6p
ELECTRONIC_STRUCTURE[55] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0]  # Cs
ELECTRONIC_STRUCTURE[56] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0]  # Ba
ELECTRONIC_STRUCTURE[57] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 1, 0]  # La
ELECTRONIC_STRUCTURE[58] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 1, 1, 0]  # Ce
ELECTRONIC_STRUCTURE[59] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 3, 0, 0]  # Pr
ELECTRONIC_STRUCTURE[60] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 4, 0, 0]  # Nd
ELECTRONIC_STRUCTURE[61] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 5, 0, 0]  # Pm
ELECTRONIC_STRUCTURE[62] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 6, 0, 0]  # Sm
ELECTRONIC_STRUCTURE[63] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 0, 0]  # Eu
ELECTRONIC_STRUCTURE[64] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 1, 0]  # Gd
ELECTRONIC_STRUCTURE[65] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 9, 0, 0]  # Tb
ELECTRONIC_STRUCTURE[66] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10, 0, 0] # Dy
ELECTRONIC_STRUCTURE[67] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11, 0, 0] # Ho
ELECTRONIC_STRUCTURE[68] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12, 0, 0] # Er
ELECTRONIC_STRUCTURE[69] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13, 0, 0] # Tm
ELECTRONIC_STRUCTURE[70] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 0, 0] # Yb
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s  4d 5p 6s  4f  5d 6p
ELECTRONIC_STRUCTURE[71] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 1, 0] # Lu
ELECTRONIC_STRUCTURE[71] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 2, 0] # Hf
ELECTRONIC_STRUCTURE[73] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 3, 0] # Ta
ELECTRONIC_STRUCTURE[74] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 4, 0] # W
ELECTRONIC_STRUCTURE[75] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 5, 0] # Re
ELECTRONIC_STRUCTURE[76] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 6, 0] # Os
ELECTRONIC_STRUCTURE[77] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 7, 0] # Ir
ELECTRONIC_STRUCTURE[78] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 9, 0] # Pt
ELECTRONIC_STRUCTURE[79] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0] # Au
ELECTRONIC_STRUCTURE[80] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0] # Hg
########################### 1s 2s 2p 3s 3p 4s 3d 4p 5s  4d 5p 6s  4f  5d 6p
ELECTRONIC_STRUCTURE[81] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1] # Tl
ELECTRONIC_STRUCTURE[82] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2] # Pb
ELECTRONIC_STRUCTURE[83] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3] # Bi
ELECTRONIC_STRUCTURE[84] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4] # Po
ELECTRONIC_STRUCTURE[85] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5] # At
ELECTRONIC_STRUCTURE[86] = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6] # Rn



VALENCE_STRUCTURE = [[0.25] * 4] * len(PERIODIC_TABLE)
###################### vs vp  vd  vf
VALENCE_STRUCTURE[1] = [1, 0, 0, 0]  # H
VALENCE_STRUCTURE[2] = [2, 0, 0, 0]  # He
VALENCE_STRUCTURE[3] = [1, 0, 0, 0]  # Li
VALENCE_STRUCTURE[4] = [2, 0, 0, 0]  # Be
VALENCE_STRUCTURE[5] = [2, 1, 0, 0]  # B
VALENCE_STRUCTURE[6] = [2, 2, 0, 0]  # C
VALENCE_STRUCTURE[7] = [2, 3, 0, 0]  # N
VALENCE_STRUCTURE[8] = [2, 4, 0, 0]  # O
VALENCE_STRUCTURE[9] = [2, 5, 0, 0]  # F
VALENCE_STRUCTURE[10] = [2, 6, 0, 0]  # Ne
VALENCE_STRUCTURE[11] = [1, 0, 0, 0]  # Na
VALENCE_STRUCTURE[12] = [2, 0, 0, 0]  # Mg
VALENCE_STRUCTURE[13] = [2, 1, 0, 0]  # Al
VALENCE_STRUCTURE[14] = [2, 2, 0, 0]  # Si
VALENCE_STRUCTURE[15] = [2, 3, 0, 0]  # P
VALENCE_STRUCTURE[16] = [2, 4, 0, 0]  # S
VALENCE_STRUCTURE[17] = [2, 5, 0, 0]  # Cl
VALENCE_STRUCTURE[18] = [2, 6, 0, 0]  # Ar
VALENCE_STRUCTURE[19] = [1, 0, 0, 0]  # K
VALENCE_STRUCTURE[20] = [2, 0, 0, 0]  # Ca
VALENCE_STRUCTURE[21] = [2, 0, 1, 0]  # Sc
VALENCE_STRUCTURE[22] = [2, 0, 2, 0]  # Ti
VALENCE_STRUCTURE[23] = [2, 0, 3, 0]  # V
VALENCE_STRUCTURE[24] = [1, 0, 5, 0]  # Cr
VALENCE_STRUCTURE[25] = [2, 0, 5, 0]  # Mn
VALENCE_STRUCTURE[26] = [2, 0, 6, 0]  # Fe
VALENCE_STRUCTURE[27] = [2, 0, 7, 0]  # Co
VALENCE_STRUCTURE[28] = [2, 0, 8, 0]  # Ni
VALENCE_STRUCTURE[29] = [1, 0, 10, 0]  # Cu
VALENCE_STRUCTURE[30] = [2, 0, 10, 0]  # Zn
VALENCE_STRUCTURE[31] = [2, 1, 10, 0]  # Ga
VALENCE_STRUCTURE[32] = [2, 2, 10, 0]  # Ge
VALENCE_STRUCTURE[33] = [2, 3, 10, 0]  # As
VALENCE_STRUCTURE[34] = [2, 4, 10, 0]  # Se
VALENCE_STRUCTURE[35] = [2, 5, 10, 0]  # Br
VALENCE_STRUCTURE[36] = [2, 6, 10, 0]  # Kr
VALENCE_STRUCTURE[37] = [1, 0, 0, 0]  # Rb
VALENCE_STRUCTURE[38] = [2, 0, 0, 0]  # Sr
VALENCE_STRUCTURE[39] = [2, 0, 1, 0]  # Y
VALENCE_STRUCTURE[40] = [2, 0, 2, 0]  # Zr
VALENCE_STRUCTURE[41] = [1, 0, 4, 0]  # Nb
VALENCE_STRUCTURE[42] = [1, 0, 5, 0]  # Mo
VALENCE_STRUCTURE[43] = [2, 0, 5, 0]  # Tc
VALENCE_STRUCTURE[44] = [1, 0, 7, 0]  # Ru
VALENCE_STRUCTURE[45] = [1, 0, 8, 0]  # Rh
VALENCE_STRUCTURE[46] = [0, 0, 10, 0]  # Pd
VALENCE_STRUCTURE[47] = [1, 0, 10, 0]  # Ag
VALENCE_STRUCTURE[48] = [2, 0, 10, 0]  # Cd
VALENCE_STRUCTURE[49] = [2, 1, 10, 0]  # In
VALENCE_STRUCTURE[50] = [2, 2, 10, 0]  # Sn
VALENCE_STRUCTURE[51] = [2, 3, 10, 0]  # Sb
VALENCE_STRUCTURE[52] = [2, 4, 10, 0]  # Te
VALENCE_STRUCTURE[53] = [2, 5, 10, 0]  # I
VALENCE_STRUCTURE[54] = [2, 6, 10, 0]  # Xe
VALENCE_STRUCTURE[55] = [1, 0, 0, 0]  # Cs
VALENCE_STRUCTURE[56] = [2, 0, 0, 0]  # Ba
VALENCE_STRUCTURE[57] = [2, 0, 1, 0]  # La
VALENCE_STRUCTURE[58] = [2, 0, 1, 1]  # Ce
VALENCE_STRUCTURE[59] = [2, 0, 0, 3]  # Pr
VALENCE_STRUCTURE[60] = [2, 0, 0, 4]  # Nd
VALENCE_STRUCTURE[61] = [2, 0, 0, 5]  # Pm
VALENCE_STRUCTURE[62] = [2, 0, 0, 6]  # Sm
VALENCE_STRUCTURE[63] = [2, 0, 0, 7]  # Eu
VALENCE_STRUCTURE[64] = [2, 0, 1, 7]  # Gd
VALENCE_STRUCTURE[65] = [2, 0, 0, 9]  # Tb
VALENCE_STRUCTURE[66] = [2, 0, 0, 10]  # Dy
VALENCE_STRUCTURE[67] = [2, 0, 0, 11]  # Ho
VALENCE_STRUCTURE[68] = [2, 0, 0, 12]  # Er
VALENCE_STRUCTURE[69] = [2, 0, 0, 13]  # Tm
VALENCE_STRUCTURE[70] = [2, 0, 0, 14]  # Yb
VALENCE_STRUCTURE[71] = [2, 0, 1, 14]  # Lu
VALENCE_STRUCTURE[72] = [2, 0, 2, 14]  # Hf
VALENCE_STRUCTURE[73] = [2, 0, 3, 14]  # Ta
VALENCE_STRUCTURE[74] = [2, 0, 4, 14]  # W
VALENCE_STRUCTURE[75] = [2, 0, 5, 14]  # Re
VALENCE_STRUCTURE[76] = [2, 0, 6, 14]  # Os
VALENCE_STRUCTURE[77] = [2, 0, 7, 14]  # Ir
VALENCE_STRUCTURE[78] = [1, 0, 9, 14]  # Pt
VALENCE_STRUCTURE[79] = [1, 0, 10, 14]  # Au
VALENCE_STRUCTURE[80] = [2, 0, 10, 14]  # Hg
VALENCE_STRUCTURE[81] = [2, 1, 10, 14]  # Tl
VALENCE_STRUCTURE[82] = [2, 2, 10, 14]  # Pb
VALENCE_STRUCTURE[83] = [2, 3, 10, 14]  # Bi
VALENCE_STRUCTURE[84] = [2, 4, 10, 14]  # Po
VALENCE_STRUCTURE[85] = [2, 5, 10, 14]  # At
VALENCE_STRUCTURE[86] = [2, 6, 10, 14]  # Rn
###################### vs vp  vd  vf

VALENCE_ELECTRONS = [ sum(e) for e in VALENCE_STRUCTURE ]

VALENCE_STRUCTURE_FULL = [[[0]*9,[0]*9]] * len(PERIODIC_TABLE)
VALENCE_STRUCTURE_FULL[1] = [[1]+[0]*8,[0]*9]  # H
VALENCE_STRUCTURE_FULL[2] = [[1]+[0]*8,[1]+[0]*8]  # He
VALENCE_STRUCTURE_FULL[3] = [[1]+[0]*8,[0]*9]  # Li
VALENCE_STRUCTURE_FULL[4] = [[1]+[0]*8,[1]+[0]*8]  # Be
VALENCE_STRUCTURE_FULL[5] = [[1]*2+[0]*7,[1]+[0]*8]  # B
VALENCE_STRUCTURE_FULL[6] = [[1]*3+[0]*6,[1]+[0]*8]  # C
VALENCE_STRUCTURE_FULL[7] = [[1]*4+[0]*5,[1]+[0]*8]  # N
VALENCE_STRUCTURE_FULL[8] = [[1]*4+[0]*5,[1]*2+[0]*7]  # O
VALENCE_STRUCTURE_FULL[9] = [[1]*4+[0]*5,[1]*3+[0]*6]  # F
VALENCE_STRUCTURE_FULL[10] = [[1]*4+[0]*5,[1]*4+[0]*5]  # Ne
VALENCE_STRUCTURE_FULL[11] = [[1]+[0]*8,[0]*9]  # Na
VALENCE_STRUCTURE_FULL[12] = [[1]+[0]*8,[1]+[0]*8]  # Mg
VALENCE_STRUCTURE_FULL[13] = [[1]*2+[0]*7,[1]+[0]*8]  # Al
VALENCE_STRUCTURE_FULL[14] = [[1]*3+[0]*6,[1]+[0]*8]  # Si
VALENCE_STRUCTURE_FULL[15] = [[1]*4+[0]*5,[1]+[0]*8]  # P
VALENCE_STRUCTURE_FULL[16] = [[1]*4+[0]*5,[1]*2+[0]*7]  # S
VALENCE_STRUCTURE_FULL[17] = [[1]*4+[0]*5,[1]*3+[0]*6]  # Cl
VALENCE_STRUCTURE_FULL[18] = [[1]*4+[0]*5,[1]*4+[0]*5]  # Ar



# ATOMIC ELECTRONEGATIVITIES FROM
# A generally applicable atomic-charge dependent London dispersion correction (SI)
# (Caldeweyher et al.)
D3_ELECTRONEGATIVITIES = (
    [1.0]
    + [
        1.23695041,
        1.26590957,
        0.54341808,
        0.99666991,
        1.26691604,
        1.40028282,
        1.55819364,
        1.56866440,
        1.57540015,
        1.15056627,
        0.55936220,
        0.72373742,
        1.12910844,
        1.12306840,
        1.52672442,
        1.40768172,
        1.48154584,
        1.31062963,
        0.40374140,
        0.75442607,
        0.76482096,
        0.98457281,
        0.96702598,
        1.05266584,
        0.93274875,
        1.04025281,
        0.92738624,
        1.07419210,
        1.07900668,
        1.04712861,
        1.15018618,
        1.15388455,
        1.36313743,
        1.36485106,
        1.39801837,
        1.18695346,
        0.36273870,
        0.58797255,
        0.71961946,
        0.96158233,
        0.89585296,
        0.81360499,
        1.00794665,
        0.92613682,
        1.09152285,
        1.14907070,
        1.13508911,
        1.08853785,
        1.11005982,
        1.12452195,
        1.21642129,
        1.36507125,
        1.40340000,
        1.16653482,
        0.34125098,
        0.58884173,
        0.68441115,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.56999999,
        0.87936784,
        1.02761808,
        0.93297476,
        1.10172128,
        0.97350071,
        1.16695666,
        1.23997927,
        1.18464453,
        1.14191734,
        1.12334192,
        1.01485321,
        1.12950808,
        1.30804834,
        1.33689961,
        1.27465977,
    ]
    + [1.0]
)

D3_HARDNESSES = (
    [1.0e3]
    + [
        -0.35015861,
        1.04121227,
        0.09281243,
        0.09412380,
        0.26629137,
        0.19408787,
        0.05317918,
        0.03151644,
        0.32275132,
        1.30996037,
        0.24206510,
        0.04147733,
        0.11634126,
        0.13155266,
        0.15350650,
        0.15250997,
        0.17523529,
        0.28774450,
        0.42937314,
        0.01896455,
        0.07179178,
        -0.01121381,
        -0.03093370,
        0.02716319,
        -0.01843812,
        -0.15270393,
        -0.09192645,
        -0.13418723,
        -0.09861139,
        0.18338109,
        0.08299615,
        0.11370033,
        0.19005278,
        0.10980677,
        0.12327841,
        0.25345554,
        0.58615231,
        0.16093861,
        0.04548530,
        -0.02478645,
        0.01909943,
        0.01402541,
        -0.03595279,
        0.01137752,
        -0.03697213,
        0.08009416,
        0.02274892,
        0.12801822,
        -0.02078702,
        0.05284319,
        0.07581190,
        0.09663758,
        0.09547417,
        0.07803344,
        0.64913257,
        0.15348654,
        0.05054344,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        0.11000000,
        -0.02786741,
        0.01057858,
        -0.03892226,
        -0.04574364,
        -0.03874080,
        -0.03782372,
        -0.07046855,
        0.09546597,
        0.21953269,
        0.02522348,
        0.15263050,
        0.08042611,
        0.01878626,
        0.08715453,
        0.10500484,
    ]
    + [1.0e3]
)

D3_KAPPA = (
    [0.0]
    + [
        0.04916110,
        0.10937243,
        -0.12349591,
        -0.02665108,
        -0.02631658,
        0.06005196,
        0.09279548,
        0.11689703,
        0.15704746,
        0.07987901,
        -0.10002962,
        -0.07712863,
        -0.02170561,
        -0.04964052,
        0.14250599,
        0.07126660,
        0.13682750,
        0.14877121,
        -0.10219289,
        -0.08979338,
        -0.08273597,
        -0.01754829,
        -0.02765460,
        -0.02558926,
        -0.08010286,
        -0.04163215,
        -0.09369631,
        -0.03774117,
        -0.05759708,
        0.02431998,
        -0.01056270,
        -0.02692862,
        0.07657769,
        0.06561608,
        0.08006749,
        0.14139200,
        -0.05351029,
        -0.06701705,
        -0.07377246,
        -0.02927768,
        -0.03867291,
        -0.06929825,
        -0.04485293,
        -0.04800824,
        -0.01484022,
        0.07917502,
        0.06619243,
        0.02434095,
        -0.01505548,
        -0.03030768,
        0.01418235,
        0.08953411,
        0.08967527,
        0.07277771,
        -0.02129476,
        -0.06188828,
        -0.06568203,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.11000000,
        -0.03585873,
        -0.03132400,
        -0.05902379,
        -0.02827592,
        -0.07606260,
        -0.02123839,
        0.03814822,
        0.02146834,
        0.01580538,
        -0.00894298,
        -0.05864876,
        -0.01817842,
        0.07721851,
        0.07936083,
        0.05849285,
    ]
    + [0.0]
)

D3_VDW_RADII = (
    [1.0]
    + [
        0.55159092,
        0.66205886,
        0.90529132,
        1.51710827,
        2.86070364,
        1.88862966,
        1.32250290,
        1.23166285,
        1.77503721,
        1.11955204,
        1.28263182,
        1.22344336,
        1.70936266,
        1.54075036,
        1.38200579,
        2.18849322,
        1.36779065,
        1.27039703,
        1.64466502,
        1.58859404,
        1.65357953,
        1.50021521,
        1.30104175,
        1.46301827,
        1.32928147,
        1.02766713,
        1.02291377,
        0.94343886,
        1.14881311,
        1.47080755,
        1.76901636,
        1.98724061,
        2.41244711,
        2.26739524,
        2.95378999,
        1.20807752,
        1.65941046,
        1.62733880,
        1.61344972,
        1.63220728,
        1.60899928,
        1.43501286,
        1.54559205,
        1.32663678,
        1.37644152,
        1.36051851,
        1.23395526,
        1.65734544,
        1.53895240,
        1.97542736,
        1.97636542,
        2.05432381,
        3.80138135,
        1.43893803,
        1.75505957,
        1.59815118,
        1.76401732,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.63999999,
        1.47055223,
        1.81127084,
        1.40189963,
        1.54015481,
        1.33721475,
        1.57165422,
        1.04815857,
        1.78342098,
        2.79106396,
        1.78160840,
        2.47588882,
        2.37670734,
        1.76613217,
        2.66172302,
        2.82773085,
    ]
    + [1.0]
)

D3_COV_RADII = (
    [1.0]
    + [
        0.32,
        0.46,
        1.33,
        1.02,
        0.85,
        0.75,
        0.71,
        0.63,
        0.64,
        0.67,
        1.55,
        1.39,
        1.26,
        1.16,
        1.11,
        1.03,
        0.99,
        0.96,
        1.96,
        1.71,
        1.48,
        1.36,
        1.34,
        1.22,
        1.19,
        1.16,
        1.11,
        1.10,
        1.12,
        1.18,
        1.24,
        1.21,
        1.21,
        1.16,
        1.14,
        1.17,
        2.10,
        1.85,
        1.63,
        1.54,
        1.47,
        1.38,
        1.28,
        1.25,
        1.25,
        1.20,
        1.28,
        1.36,
        1.42,
        1.40,
        1.40,
        1.36,
        1.33,
        1.31,
        2.32,
        1.96,
        1.80,
        1.63,
        1.76,
        1.74,
        1.73,
        1.72,
        1.68,
        1.69,
        1.68,
        1.67,
        1.66,
        1.65,
        1.64,
        1.70,
        1.62,
        1.52,
        1.46,
        1.37,
        1.31,
        1.29,
        1.22,
        1.23,
        1.24,
        1.33,
        1.44,
        1.44,
        1.51,
        1.45,
        1.47,
        1.42,
    ]
    + [1.0]
)
D3_COV_RADII = [x / au.BOHR for x in D3_COV_RADII]

# free atom vdw radii in bohr from Tkatchenko-Scheffler
VDW_RADII = [
    1.0,  # Dummy
    3.1,  # H
    2.65,  # He
    4.16,  # Li
    4.17,  # Be
    3.89,  # B
    3.59,  # C
    3.34,  # N
    3.19,  # O
    3.04,  # F
    2.91,  # Ne
    3.73,  # Na
    4.27,  # Mg
    4.33,  # Al
    4.2,  # Si
    4.01,  # P
    3.86,  # S
    3.71,  # Cl
    3.55,  # Ar
    3.71,  # K
    4.65,  # Ca
    4.59,  # Sc
    4.51,  # Ti
    4.44,  # V
    3.99,  # Cr
    3.97,  # Mn
    4.23,  # Fe
    4.18,  # Co
    3.82,  # Ni
    3.76,  # Cu
    4.02,  # Zn
    4.19,  # Ga
    4.2,  # Ge
    4.11,  # As
    4.04,  # Se
    3.93,  # Br
    3.82,  # Kr
    3.72,  # Rb
    4.54,  # Sr
    4.8151,  # Y
    4.53,  # Zr
    4.2365,  # Nb
    4.099,  # Mo
    4.076,  # Tc
    3.9953,  # Ru
    3.95,  # Rh
    3.66,  # Pd
    3.82,  # Ag
    3.99,  # Cd
    4.23198,  # In
    4.303,  # Sn
    4.276,  # Sb
    4.22,  # Te
    4.17,  # I
    4.08,  # Xe
    3.78,  # Cs
    4.77,  # Ba
    3.14,  # La
    3.26,  # Ce
    3.28,  # Pr
    3.3,  # Nd
    3.27,  # Pm
    3.32,  # Sm
    3.4,  # Eu
    3.62,  # Gd
    3.42,  # Tb
    3.26,  # Dy
    3.24,  # Ho
    3.3,  # Er
    3.26,  # Tm
    3.22,  # Yb
    3.2,  # Lu
    4.21,  # Hf
    4.15,  # Ta
    4.08,  # W
    4.02,  # Re
    3.84,  # Os
    4.0,  # Ir
    3.92,  # Pt
    3.86,  # Au
    3.98,  # Hg
    3.91,  # Tl
    4.31,  # Pb
    4.32,  # Bi
    4.097,  # Po
    4.07,  # At
    4.23,  # Rn
    3.9,  # Fr
    4.98,  # Ra
    2.75,  # Ac
    2.85,  # Th
    2.71,  # Pa
    3.0,  # U
    3.28,  # Np
    3.45,  # Pu
    3.51,  # Am
    3.47,  # Cm
    3.56,  # Bk
    3.55,  # Cf
    3.76,  # Es
    3.89,  # Fm
    3.93,  # Md
    3.78,  # No
    1.0,  # Dummy
]

# free atom C6 coefficients in hartree*bohr**6
C6_FREE = [
    0.0,  # Dummy
    6.5,  # H
    1.46,  # He
    1387.0,  # Li
    214.0,  # Be
    99.5,  # B
    46.6,  # C
    24.2,  # N
    15.6,  # O
    9.52,  # F
    6.38,  # Ne
    1556.0,  # Na
    627.0,  # Mg
    528.0,  # Al
    305.0,  # Si
    185.0,  # P
    134.0,  # S
    94.6,  # Cl
    64.3,  # Ar
    3897.0,  # K
    2221.0,  # Ca
    1383.0,  # Sc
    1044.0,  # Ti
    832.0,  # V
    602.0,  # Cr
    552.0,  # Mn
    482.0,  # Fe
    408.0,  # Co
    373.0,  # Ni
    253.0,  # Cu
    284.0,  # Zn
    498.0,  # Ga
    354.0,  # Ge
    246.0,  # As
    210.0,  # Se
    162.0,  # Br
    129.6,  # Kr
    4691.0,  # Rb
    3170.0,  # Sr
    1968.58,  # Y
    1677.91,  # Zr
    1263.61,  # Nb
    1028.73,  # Mo
    1390.87,  # Tc
    609.754,  # Ru
    469.0,  # Rh
    157.5,  # Pd
    339.0,  # Ag
    452.0,  # Cd
    707.046,  # In
    587.417,  # Sn
    459.322,  # Sb
    396.0,  # Te
    385.0,  # I
    285.9,  # Xe
    6582.08,  # Cs
    5727.0,  # Ba
    3884.5,  # La
    3708.33,  # Ce
    3911.84,  # Pr
    3908.75,  # Nd
    3847.68,  # Pm
    3708.69,  # Sm
    3511.71,  # Eu
    2781.53,  # Gd
    3124.41,  # Tb
    2984.29,  # Dy
    2839.95,  # Ho
    2724.12,  # Er
    2576.78,  # Tm
    2387.53,  # Yb
    2371.8,  # Lu
    1274.8,  # Hf
    1019.92,  # Ta
    847.93,  # W
    710.2,  # Re
    596.67,  # Os
    359.1,  # Ir
    347.1,  # Pt
    298.0,  # Au
    392.0,  # Hg
    717.44,  # Tl
    697.0,  # Pb
    571.0,  # Bi
    530.92,  # Po
    457.53,  # At
    420.6,  # 390.63,  #Rn
    4224.44,  # Fr
    4851.32,  # Ra
    3604.41,  # Ac
    4047.54,  # Th
    2367.42,  # Pa
    1877.1,  # U
    2507.88,  # Np
    2117.27,  # Pu
    2110.98,  # Am
    2403.22,  # Cm
    1985.82,  # Bk
    1891.92,  # Cf
    1851.1,  # Es
    1787.07,  # Fm
    1701.0,  # Md
    1578.18,  # No
    0.0,  # Dummy
]

# free atom polarizabilities in bohr**3
POLARIZABILITIES = [
    1.0e-6,  # Dummy
    4.5,  # H
    1.38,  # He
    164.2,  # Li
    38.0,  # Be
    21.0,  # B
    12.0,  # C
    7.4,  # N
    5.4,  # O
    3.8,  # F
    2.67,  # Ne
    162.7,  # Na
    71.0,  # Mg
    60.0,  # Al
    37.0,  # Si
    25.0,  # P
    19.6,  # S
    15.0,  # Cl
    11.1,  # Ar
    292.9,  # K
    160.0,  # Ca
    120.0,  # Sc
    98.0,  # Ti
    84.0,  # V
    78.0,  # Cr
    63.0,  # Mn
    56.0,  # Fe
    50.0,  # Co
    48.0,  # Ni
    42.0,  # Cu
    40.0,  # Zn
    60.0,  # Ga
    41.0,  # Ge
    29.0,  # As
    25.0,  # Se
    20.0,  # Br
    16.8,  # Kr
    319.2,  # Rb
    199.0,  # Sr
    126.737,  # Y
    119.97,  # Zr
    101.603,  # Nb
    88.4225785,  # Mo
    80.083,  # Tc
    65.895,  # Ru
    56.1,  # Rh
    23.68,  # Pd
    50.6,  # Ag
    39.7,  # Cd
    70.22,  # In
    55.95,  # Sn
    43.67197,  # Sb
    37.65,  # Te
    35.0,  # I
    27.3,  # Xe
    427.12,  # Cs
    275.0,  # Ba
    213.7,  # La
    204.7,  # Ce
    215.8,  # Pr
    208.4,  # Nd
    200.2,  # Pm
    192.1,  # Sm
    184.2,  # Eu
    158.3,  # Gd
    169.5,  # Tb
    164.64,  # Dy
    156.3,  # Ho
    150.2,  # Er
    144.3,  # Tm
    138.9,  # Yb
    137.2,  # Lu
    99.52,  # Hf
    82.53,  # Ta
    71.041,  # W
    63.04,  # Re
    55.055,  # Os
    42.51,  # Ir
    39.68,  # Pt
    36.5,  # Au
    33.9,  # Hg
    69.92,  # Tl
    61.8,  # Pb
    49.02,  # Bi
    45.013,  # Po
    38.93,  # At
    33.54,  # Rn
    317.8,  # Fr
    246.2,  # Ra
    203.3,  # Ac
    217.0,  # Th
    154.4,  # Pa
    127.8,  # U
    150.5,  # Np
    132.2,  # Pu
    131.2,  # Am
    143.6,  # Cm
    125.3,  # Bk
    121.5,  # Cf
    117.5,  # Es
    113.4,  # Fm
    109.4,  # Md
    105.4,  # No
    1.0e-6,  # Dummy
]

PAULING_ELECTRONEGATIVITY = [0.0] * len(PERIODIC_TABLE)
PAULING_ELECTRONEGATIVITY[1] = 2.20
PAULING_ELECTRONEGATIVITY[2] = 4.42  # -1. organov
PAULING_ELECTRONEGATIVITY[3] = 0.98
PAULING_ELECTRONEGATIVITY[4] = 1.57
PAULING_ELECTRONEGATIVITY[5] = 2.04
PAULING_ELECTRONEGATIVITY[6] = 2.55
PAULING_ELECTRONEGATIVITY[7] = 3.04
PAULING_ELECTRONEGATIVITY[8] = 3.44
PAULING_ELECTRONEGATIVITY[9] = 3.98
PAULING_ELECTRONEGATIVITY[10] = 4.44  # -1 organov
PAULING_ELECTRONEGATIVITY[11] = 0.93
PAULING_ELECTRONEGATIVITY[12] = 1.31
PAULING_ELECTRONEGATIVITY[13] = 1.61
PAULING_ELECTRONEGATIVITY[14] = 1.90
PAULING_ELECTRONEGATIVITY[15] = 2.19
PAULING_ELECTRONEGATIVITY[16] = 2.58
PAULING_ELECTRONEGATIVITY[17] = 3.16
PAULING_ELECTRONEGATIVITY[18] = 3.57  # -1 organov
PAULING_ELECTRONEGATIVITY[19] = 0.82
PAULING_ELECTRONEGATIVITY[20] = 1.00
PAULING_ELECTRONEGATIVITY[21] = 1.36
PAULING_ELECTRONEGATIVITY[22] = 1.54
PAULING_ELECTRONEGATIVITY[23] = 1.63
PAULING_ELECTRONEGATIVITY[24] = 1.66
PAULING_ELECTRONEGATIVITY[25] = 1.55
PAULING_ELECTRONEGATIVITY[26] = 1.83
PAULING_ELECTRONEGATIVITY[27] = 1.88
PAULING_ELECTRONEGATIVITY[28] = 1.91
PAULING_ELECTRONEGATIVITY[29] = 1.90
PAULING_ELECTRONEGATIVITY[30] = 1.65
PAULING_ELECTRONEGATIVITY[31] = 1.81
PAULING_ELECTRONEGATIVITY[32] = 2.01
PAULING_ELECTRONEGATIVITY[33] = 2.18
PAULING_ELECTRONEGATIVITY[34] = 2.55
PAULING_ELECTRONEGATIVITY[35] = 2.96
PAULING_ELECTRONEGATIVITY[36] = 3.37  # -1 organov
PAULING_ELECTRONEGATIVITY[37] = 0.82
PAULING_ELECTRONEGATIVITY[38] = 0.95
PAULING_ELECTRONEGATIVITY[39] = 1.22
PAULING_ELECTRONEGATIVITY[40] = 1.33
PAULING_ELECTRONEGATIVITY[41] = 1.60
PAULING_ELECTRONEGATIVITY[42] = 2.16
PAULING_ELECTRONEGATIVITY[43] = 1.90
PAULING_ELECTRONEGATIVITY[44] = 2.20
PAULING_ELECTRONEGATIVITY[45] = 2.28
PAULING_ELECTRONEGATIVITY[46] = 2.20
PAULING_ELECTRONEGATIVITY[47] = 1.93
PAULING_ELECTRONEGATIVITY[48] = 1.69
PAULING_ELECTRONEGATIVITY[49] = 1.78
PAULING_ELECTRONEGATIVITY[50] = 1.96
PAULING_ELECTRONEGATIVITY[51] = 2.05
PAULING_ELECTRONEGATIVITY[52] = 2.10
PAULING_ELECTRONEGATIVITY[53] = 2.66
PAULING_ELECTRONEGATIVITY[54] = 3.12  # -1 organov

SJS_COORDINATES = [[0] * 4] * len(PERIODIC_TABLE)
SJS_COORDINATES[1] = [0, 1, 1, 0]
SJS_COORDINATES[2] = [0, 1, -1, 0]
SJS_COORDINATES[3] = [0, 2, 1, 0]
SJS_COORDINATES[4] = [0, 2, -1, 0]
SJS_COORDINATES[5] = [1, 3, 1, -1]
SJS_COORDINATES[6] = [1, 3, 2, 0]
SJS_COORDINATES[7] = [1, 3, 1, 1]
SJS_COORDINATES[8] = [1, 3, -1, -1]
SJS_COORDINATES[9] = [1, 3, -2, 0]
SJS_COORDINATES[10] = [1, 3, -1, 1]
SJS_COORDINATES[11] = [0, 3, 1, 0]
SJS_COORDINATES[12] = [0, 3, -1, 0]
SJS_COORDINATES[13] = [1, 4, 1, -1]
SJS_COORDINATES[14] = [1, 4, 2, 0]
SJS_COORDINATES[15] = [1, 4, 1, 1]
SJS_COORDINATES[16] = [1, 4, -1, -1]
SJS_COORDINATES[17] = [1, 4, -2, 0]
SJS_COORDINATES[18] = [1, 4, -1, 1]
SJS_COORDINATES[19] = [0, 4, 1, 0]
SJS_COORDINATES[20] = [0, 4, -1, 0]
SJS_COORDINATES[21] = [2, 5, 1, -2]
SJS_COORDINATES[22] = [2, 5, 2, -1]
SJS_COORDINATES[23] = [2, 5, 3, 0]
SJS_COORDINATES[24] = [2, 5, 2, 1]
SJS_COORDINATES[25] = [2, 5, 1, 2]
SJS_COORDINATES[26] = [2, 5, -1, -2]
SJS_COORDINATES[27] = [2, 5, -2, -1]
SJS_COORDINATES[28] = [2, 5, -3, 0]
SJS_COORDINATES[29] = [2, 5, -2, 1]
SJS_COORDINATES[30] = [2, 5, -1, 2]
SJS_COORDINATES[31] = [1, 5, 1, -1]
SJS_COORDINATES[32] = [1, 5, 2, 0]
SJS_COORDINATES[33] = [1, 5, 1, 1]
SJS_COORDINATES[34] = [1, 5, -1, -1]
SJS_COORDINATES[35] = [1, 5, -2, 0]
SJS_COORDINATES[36] = [1, 5, -1, 1]
SJS_COORDINATES[37] = [0, 5, 1, 0]
SJS_COORDINATES[38] = [0, 5, -1, 0]
SJS_COORDINATES[39] = [2, 6, 1, -2]
SJS_COORDINATES[40] = [2, 6, 2, -1]
SJS_COORDINATES[41] = [2, 6, 3, 0]
SJS_COORDINATES[42] = [2, 6, 2, 1]
SJS_COORDINATES[43] = [2, 6, 1, 2]
SJS_COORDINATES[44] = [2, 6, -1, -2]
SJS_COORDINATES[45] = [2, 6, -2, -1]
SJS_COORDINATES[46] = [2, 6, -3, 0]
SJS_COORDINATES[47] = [2, 6, -2, 1]
SJS_COORDINATES[48] = [2, 6, -1, 2]
SJS_COORDINATES[49] = [1, 6, 1, -1]
SJS_COORDINATES[50] = [1, 6, 2, 0]
SJS_COORDINATES[51] = [1, 6, 1, 1]
SJS_COORDINATES[52] = [1, 6, -1, -1]
SJS_COORDINATES[53] = [1, 6, -2, 0]
SJS_COORDINATES[54] = [1, 6, -1, 1]
SJS_COORDINATES[55] = [0, 6, 1, 0]
SJS_COORDINATES[56] = [0, 6, -1, 0]
SJS_COORDINATES[57] = [2, 7, 1, -2]
SJS_COORDINATES[58] = [3, 7, 1, -3]
SJS_COORDINATES[59] = [3, 7, 2, -2]
SJS_COORDINATES[60] = [3, 7, 3, -1]
SJS_COORDINATES[61] = [3, 7, 4, 0]
SJS_COORDINATES[62] = [3, 7, 3, 1]
SJS_COORDINATES[63] = [3, 7, 2, 2]
SJS_COORDINATES[64] = [3, 7, 1, 3]
SJS_COORDINATES[65] = [3, 7, -1, -3]
SJS_COORDINATES[66] = [3, 7, -2, -2]
SJS_COORDINATES[67] = [3, 7, -3, -1]
SJS_COORDINATES[68] = [3, 7, -4, 0]
SJS_COORDINATES[69] = [3, 7, -3, 1]
SJS_COORDINATES[70] = [3, 7, -2, 2]
SJS_COORDINATES[71] = [3, 7, -1, 3]
SJS_COORDINATES[72] = [2, 7, 2, -1]
SJS_COORDINATES[73] = [2, 7, 3, 0]
SJS_COORDINATES[74] = [2, 7, 2, 1]
SJS_COORDINATES[75] = [2, 7, 1, 2]
SJS_COORDINATES[76] = [2, 7, -1, -2]
SJS_COORDINATES[77] = [2, 7, -2, -1]
SJS_COORDINATES[78] = [2, 7, -3, 0]
SJS_COORDINATES[79] = [2, 7, -2, 1]
SJS_COORDINATES[80] = [2, 7, -1, 2]
SJS_COORDINATES[81] = [1, 7, 1, -1]
SJS_COORDINATES[82] = [1, 7, 2, 0]
SJS_COORDINATES[83] = [1, 7, 1, 1]
SJS_COORDINATES[84] = [1, 7, -1, -1]
SJS_COORDINATES[85] = [1, 7, -2, 0]
SJS_COORDINATES[86] = [1, 7, -1, 1]
SJS_COORDINATES[87] = [0, 7, 1, 0]
SJS_COORDINATES[88] = [0, 7, -1, 0]
SJS_COORDINATES[89] = [2, 8, 1, -2]
SJS_COORDINATES[90] = [3, 8, 1, -3]
SJS_COORDINATES[91] = [3, 8, 2, -2]
SJS_COORDINATES[92] = [3, 8, 3, -1]
SJS_COORDINATES[93] = [3, 8, 4, 0]
SJS_COORDINATES[94] = [3, 8, 3, 1]
SJS_COORDINATES[95] = [3, 8, 2, 2]
SJS_COORDINATES[96] = [3, 8, 1, 3]
SJS_COORDINATES[97] = [3, 8, -1, -3]
SJS_COORDINATES[98] = [3, 8, -2, -2]
SJS_COORDINATES[99] = [3, 8, -3, -1]
SJS_COORDINATES[100] = [3, 8, -4, 0]
SJS_COORDINATES[101] = [3, 8, -3, 1]
SJS_COORDINATES[102] = [3, 8, -2, 2]
SJS_COORDINATES[103] = [3, 8, -1, 3]
SJS_COORDINATES[104] = [2, 8, 2, -1]
SJS_COORDINATES[105] = [2, 8, 3, 0]
SJS_COORDINATES[106] = [2, 8, 2, 1]
SJS_COORDINATES[107] = [2, 8, 1, 2]
SJS_COORDINATES[108] = [2, 8, -1, -2]
SJS_COORDINATES[109] = [2, 8, -2, -1]
SJS_COORDINATES[110] = [2, 8, -3, 0]
SJS_COORDINATES[111] = [2, 8, -2, 1]
SJS_COORDINATES[112] = [2, 8, -1, 2]
SJS_COORDINATES[113] = [1, 8, 1, -1]
SJS_COORDINATES[114] = [1, 8, 2, 0]
SJS_COORDINATES[115] = [1, 8, 1, 1]
SJS_COORDINATES[116] = [1, 8, -1, -1]
SJS_COORDINATES[117] = [1, 8, -2, 0]
SJS_COORDINATES[118] = [1, 8, -1, 1]

ATOMIC_IONIZATION_ENERGY=[0.]*len(PERIODIC_TABLE)
ATOMIC_IONIZATION_ENERGY[1]=1312/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[2]=2372/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[3]=513/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[4]=899/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[5]=801/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[6]=1086/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[7]=1402/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[8]=1314/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[9]=1681/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[10]=2081/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[11]=496/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[12]=738/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[13]=577/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[14]=786/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[15]=1012/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[16]=1000/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[17]=1251/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[18]=1520/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[19]=419/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[20]=590/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[21]=631/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[22]=658/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[23]=650/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[24]=653/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[25]=717/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[26]=759/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[27]=760/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[28]=737/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[29]=745/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[30]=906/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[31]=579/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[32]=762/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[33]=947/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[34]=941/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[35]=1140/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[36]=1351/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[37]=403/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[38]=550/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[39]=616/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[40]=660/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[41]=664/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[42]=685/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[43]=702/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[44]=711/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[45]=720/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[46]=805/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[47]=731/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[48]=868/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[49]=558/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[50]=709/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[51]=834/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[52]=869/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[53]=1008/au.KJPERMOL
ATOMIC_IONIZATION_ENERGY[54]=1170/au.KJPERMOL

ATOMIC_ELECTRON_AFFINITY=[0.]*len(PERIODIC_TABLE)
ATOMIC_ELECTRON_AFFINITY[1]=72.769/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[2]=-48/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[3]=59.632/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[4]=-48/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[5]=26.989/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[6]=121.776/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[7]=-6.8/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[8]=140.975/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[9]=328.164/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[10]=-116/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[11]=52.867/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[12]=-40/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[13]=41.762/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[14]=134.068/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[15]=72.037/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[16]=200.410/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[17]=348.575/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[18]=-96/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[19]=48.383/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[20]=2.37/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[21]=17.307/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[22]=7.289/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[23]=50.911/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[24]=65.217/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[25]=-50/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[26]=14.785/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[27]=63.897/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[28]=111.65/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[29]=119.235/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[30]=-58/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[31]=29.058/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[32]=118.935/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[33]=77.65/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[34]=194.958/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[35]=324.536/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[36]=-96/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[37]=46.884/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[38]=5.023/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[39]=30.035/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[40]=41.806/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[41]=88.516/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[42]=72.097/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[43]=53/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[44]=100.950/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[45]=110.27/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[46]=54.24/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[47]=125.862/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[48]=-68/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[49]=37.043/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[50]=107.298/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[51]=101.059/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[52]=190.161/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[53]=295.153/au.KJPERMOL
ATOMIC_ELECTRON_AFFINITY[54]=295.153/au.KJPERMOL


CHEMICAL_PROPERTIES = {
    "ATOMIC_NUMBER": [i for i in range(len(PERIODIC_TABLE))],
    "ATOMIC_MASSES": ATOMIC_MASSES,
    "ELECTRONIC_STRUCTURE": ELECTRONIC_STRUCTURE,
    "VALENCE_STRUCTURE": VALENCE_STRUCTURE,
    "VALENCE_ELECTRONS": VALENCE_ELECTRONS,
    "D3_ELECTRONEGATIVITIES": D3_ELECTRONEGATIVITIES,
    "D3_HARDNESSES": D3_HARDNESSES,
    "D3_KAPPA": D3_KAPPA,
    "D3_VDW_RADII": D3_VDW_RADII,
    "D3_COV_RADII": D3_COV_RADII,
    "VDW_RADII": VDW_RADII,
    "C6_FREE": C6_FREE,
    "POLARIZABILITIES": POLARIZABILITIES,
    "PAULING_ELECTRONEGATIVITY": PAULING_ELECTRONEGATIVITY,
    "SJS_COORDINATES": SJS_COORDINATES,
}
