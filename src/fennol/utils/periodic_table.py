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

ATOMIC_MASSES=[0.        ,   1.008     ,   4.002602  ,   6.94      , # noqa
             9.0121831 ,  10.81      ,  12.011     ,  14.007     , # noqa
            15.999     ,  18.99840316,  20.1797    ,  22.98976928, # noqa
            24.305     ,  26.9815385 ,  28.085     ,  30.973762  , # noqa
            32.06      ,  35.45      ,  39.948     ,  39.0983    , # noqa
            40.078     ,  44.955908  ,  47.867     ,  50.9415    , # noqa
            51.9961    ,  54.938044  ,  55.845     ,  58.933194  , # noqa
            58.6934    ,  63.546     ,  65.38      ,  69.723     , # noqa
            72.63      ,  74.921595  ,  78.971     ,  79.904     , # noqa
            83.798     ,  85.4678    ,  87.62      ,  88.90584   , # noqa
            91.224     ,  92.90637   ,  95.95      ,  97.90721   , # noqa
           101.07      , 102.9055    , 106.42      , 107.8682    , # noqa
           112.414     , 114.818     , 118.71      , 121.76      , # noqa
           127.6       , 126.90447   , 131.293     , 132.90545196, # noqa
           137.327     , 138.90547   , 140.116     , 140.90766   , # noqa
           144.242     , 144.91276   , 150.36      , 151.964     , # noqa
           157.25      , 158.92535   , 162.5       , 164.93033   , # noqa
           167.259     , 168.93422   , 173.054     , 174.9668    , # noqa
           178.49      , 180.94788   , 183.84      , 186.207     , # noqa
           190.23      , 192.217     , 195.084     , 196.966569  , # noqa
           200.592     , 204.38      , 207.2       , 208.9804    , # noqa
           208.98243   , 209.98715   , 222.01758   , 223.01974   , # noqa
           226.02541   , 227.02775   , 232.0377    , 231.03588   , # noqa
           238.02891   , 237.04817   , 244.06421   , 243.06138   , # noqa
           247.07035   , 247.07031   , 251.07959   , 252.083     , # noqa
           257.09511   , 258.09843   , 259.101     , 262.11      , # noqa
           267.122     , 268.126     , 271.134     , 270.133     , # noqa
           269.1338    , 278.156     , 281.165     , 281.166     , # noqa
           285.177     , 286.182     , 289.19      , 289.194     , # noqa
           293.204     , 293.208     , 294.214]

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