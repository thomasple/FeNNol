import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable
import numpy as np

PERIODIC_TABLE = (
    ["Dummy"]
    + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
)

PERIODIC_TABLE_REV_IDX = {s: i for i, s in enumerate(PERIODIC_TABLE)}

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

        conv_tensor = [-1]*(maxidx + 2)
        for i, s in enumerate(self.species_order):
            conv_tensor[rev_idx[s]] = i

        self.conv_tensor=self.variable(
            'buffers', 'conv_tensor',
            jnp.array(conv_tensor, dtype=jnp.int32)
        )
        
    @nn.compact
    def __call__(self, species):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return converted_species.to(species.device)

    def __repr__(self):
        return f'SpeciesConverter({self.species})'  