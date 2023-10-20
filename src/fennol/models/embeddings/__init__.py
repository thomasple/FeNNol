from .gaussian_moments import GaussianMomentsEmbedding
from .ani import ANIAEV
from .allegro import AllegroEmbedding,AllegroE3NNEmbedding


EMBEDDINGS={
    "GAUSSIAN_MOMENTS_EMBEDDING": GaussianMomentsEmbedding,
    "ANI_AEV": ANIAEV,
    "ALLEGRO": AllegroEmbedding,
    "ALLEGRO_E3NN": AllegroE3NNEmbedding,
}