from ..fold import cosine_fold_generator
from .linear import LinearFaultGenerator


def linear_fault_generator(shape,velseed, max_nfaults=2):
    fold = cosine_fold_generator(shape,velseed)
    fault = LinearFaultGenerator(fold, max_nfaults)
    return fault

