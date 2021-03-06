import numpy as np
from .model import Random


class CosineFold:
    def __init__(self, amax=0.05, hmax=0.05, uniform=True, first=1, dtype=np.int32, random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.amax=amax
        self.hmax=hmax
        self.dtype=dtype
        self.vround=vround
        self.uniform = uniform
        self.first = first

    def _gen_cosine(self,nx,ny):
        a1 = self.random.uniform(low=1, high=self.amax * ny)
        h1 = self.random.uniform(low=1, high=self.hmax * nx)
        r1 = self.random.array(0,1,nx,sort=True)
        return np.round(a1 * np.cos(h1 * np.pi * r1),self.vround).astype(self.dtype)

    def check_interface(self,model):
        model.interface[model.interface < 0] = 0
        for i in range(model.nlayers):
            diff = model.interface[i+1] - model.interface[i]
            if diff.min() < 0:
                model.interface[i] -= diff.min()

    def generate(self, model):
        nx,ny = model.shape
        fold = self._gen_cosine(nx,ny)
        for i in range(self.first, model.nlayers):
            model.interface[i] += fold
            if not self.uniform:
                fold = self._gen_cosine(nx,ny)
        self.check_interface(model)
        model.set_interface(model.interface.astype(self.dtype))
        return model


