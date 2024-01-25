from .gaussian_moments import GaussianMomentsEmbedding
from .ani import ANIAEV
from .allegro import AllegroEmbedding,AllegroE3NNEmbedding
from .hipnn import HIPNNEmbedding
from .newtonnet import NewtonNetEmbedding
from .painn import PAINNEmbedding
from .eeacsf import EEACSF
from .caiman import CaimanEmbedding
from .spookynet import SpookyNetEmbedding
from .foam import FOAMEmbedding
from .chgnet import CHGNetEmbedding
from .deeppot import DeepPotEmbedding,DeepPotE3Embedding
EMBEDDINGS={
    "GAUSSIAN_MOMENTS_EMBEDDING": GaussianMomentsEmbedding,
    "ANI_AEV": ANIAEV,
    "ALLEGRO": AllegroEmbedding,
    "ALLEGRO_E3NN": AllegroE3NNEmbedding,
    "HIPNN": HIPNNEmbedding,
    "NEWTONNET": NewtonNetEmbedding,
    "PAINN": PAINNEmbedding,
    "SPOOKYNET": SpookyNetEmbedding,
    "EEACSF": EEACSF,
    "CAIMAN": CaimanEmbedding,
    "FOAM": FOAMEmbedding,
    "CHGNET": CHGNetEmbedding,
    "DEEPPOT": DeepPotEmbedding,
    "DEEPPOT_E3": DeepPotE3Embedding,
}