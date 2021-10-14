from ..fold import cosine_fold_generator
from .gaussian import GaussianSaltGenerator


def gaussian_salt_generator(shape,velseed,vel_salt=4.5):
    fold = cosine_fold_generator(shape,velseed)
    salt = GaussianSaltGenerator(fold,vel_salt=vel_salt)
    return salt

