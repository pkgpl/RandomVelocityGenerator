from ..flat import flat_generator
from .cosine import CosineFoldGenerator


def cosine_fold_generator(shape,velseed):
    flat = flat_generator(shape,velseed)
    fold = CosineFoldGenerator(flat)
    return fold

