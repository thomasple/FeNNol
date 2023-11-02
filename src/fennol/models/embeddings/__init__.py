from .gaussian_moments import GaussianMomentsEmbedding
from .ani import ANIAEV
from .allegro import AllegroEmbedding,AllegroE3NNEmbedding
from .hipnn import HIPNNEmbedding
from .newtonnet import NewtonNetEmbedding
from .painn import PAINNEmbedding
from .eeacsf import EEACSF


EMBEDDINGS={
    "GAUSSIAN_MOMENTS_EMBEDDING": GaussianMomentsEmbedding,
    "ANI_AEV": ANIAEV,
    "ALLEGRO": AllegroEmbedding,
    "ALLEGRO_E3NN": AllegroE3NNEmbedding,
    "HIPNN": HIPNNEmbedding,
    "NEWTONNET": NewtonNetEmbedding,
    "PAINN": PAINNEmbedding,
    "EEACSF": EEACSF,
}