import numpy as np
from .model import Random, Model, Pipeline
from .layer import FlatLayer, DippingLayer, LinearWaterLayer
from .fold import CosineFold
from .fault import LinearFault
from .salt import GaussianSalt, EllipticSalt


def flat_generator(shape, velseed):
    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer()
        ], model)
    return pipe

def dip_generator(shape, velseed):
    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer()
        ], model)
    return pipe

def cosine_fold_generator(shape, velseed):
    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        CosineFold()
        ], model)
    return pipe

def linear_fault_generator(shape, velseed, max_nfaults=2):
    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        CosineFold(),
        LinearFault(max_nfaults=max_nfaults)
        ], model)
    return pipe

def gaussian_salt_generator(shape, velseed, vsalt=4.5):
    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        CosineFold(),
        GaussianSalt(vsalt=vsalt)
        ], model)
    return pipe

def elliptic_salt_generator(shape, velseed, vsalt=4.5, nsalts=2):
    model = Model(shape, velseed)
    steps = [FlatLayer(), CosineFold()] + [EllipticSalt(vsalt=vsalt)] * nsalts
    pipe = Pipeline(steps, model)
    return pipe

def gaussian_elliptic_salt_generator(shape, velseed, vsalt=4.5):
    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        CosineFold(),
        GaussianSalt(vsalt=vsalt),
        EllipticSalt(vsalt=vsalt)
        ], model)
    return pipe

def gom_generator(shape, dz, velseed=None, vsalt=4.5, v0=1.5, vwater=1.5, nlayers=None, k=None, generate=False):
    random = Random()
    zmax = shape[1]*dz
    if velseed is None:
        # GOM k=0.4
        _k = k or random.uniform(low=0.38, high=0.42)
        _nlayers = nlayers or random.uniform(low=10, high=25, dtype=np.int32)
        _velseed = np.linspace(v0, v0 + _k * zmax, _nlayers)
    else:
        _velseed = velseed
    model = Model(shape, _velseed)

    _vsalt = vsalt or random.uniform(low=4.5, high=5.0)
    # salt type
    salt1 = [GaussianSalt(vsalt=_vsalt, width_range=(0.05,0.1), height_range=(0.4,0.6))]
    salt2 = salt1 * 2
    salt3 = [EllipticSalt(vsalt=_vsalt)]
    salt4 = salt3 * 2
    salt5 = salt1 + salt3
    salts = [salt1, salt2, salt3, salt4, salt5]

    isalt = random.choice(np.arange(len(salts),dtype=np.int32))
    salt = salts[isalt]

    steps = [DippingLayer(y_range=(0.1,0.9), minsplit=0.01)] + salt + [LinearWaterLayer(vwater=vwater, y_range=(0.1,0.2))]
    pipe = Pipeline(steps, model)
    if generate:
        return pipe.generate()
    else:
        return pipe
