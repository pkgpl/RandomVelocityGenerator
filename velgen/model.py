import numpy as np
import sys
from collections import defaultdict

def errexit(msg):
    print(msg)
    sys.exit(1)


class Random:
    def __init__(self, random_seed=None, vround=4):
        self.vround=vround
        self.rng = np.random.default_rng(random_seed)
        self.choice = self.rng.choice

    def round(self,arr):
        return np.round(arr, self.vround)

    def uniform(self, low, high, size=None, dtype=np.float32):
        return self.round(self.rng.uniform(low=low, high=high, size=size)).astype(dtype)

    def array(self, low, high, size, sort=True, prepend=None, append=None, dtype=np.float32):
        arr = self.uniform(low=low,high=high,size=size)
        if sort:
            arr.sort()
        if prepend is not None:
            arr = np.insert(arr, 0, prepend)
        if append is not None:
            arr = np.append(arr, append)
        return self.round(arr).astype(dtype)

    def array_interval(self, low, high, size, minsplit, prepend=None, append=None, dtype=np.float32):
        sort = True
        layer = self.array(low,high,size,sort,prepend,append,dtype)
        diff = layer[1:] - layer[:-1]
        self.count=1
        while diff.min() < minsplit:
            layer = self.array(low,high,size,sort,prepend,append,dtype)
            diff = layer[1:] - layer[:-1]
            self.count += 1
        return layer

    def perturb(self, arr, max_pert, fix_top=False, fix_bottom=False):
        arr_copy = arr.copy()
        arr += self.uniform(low=-max_pert, high=max_pert, size=arr.shape)
        if fix_top:
            arr[0] = arr_copy[0]
        if fix_bottom:
            arr[-1] = arr_copy[-1]
        return self.round(arr)


class Model:
    def __init__(self, shape, velseed, max_pert=0.1, random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.max_pert = max_pert
        self.shape = shape
        self.nx, self.ny = shape
        self.layer = None
        self.velseed = np.array(velseed)
        if self.velseed.ndim == 1:
            self.veltype = 'constant'
        elif self.velseed.ndim == 2:
            self.veltype = 'linear'
        self.nlayers = len(self.velseed)
        self.velocity = np.zeros(self.shape, dtype=np.float32)
        self.filled = False
        self.history=defaultdict(list)

    def fill_const_velocity(self,velseed):
        for ix in range(self.nx):
            for i in range(self.nlayers):
                iy0 = self.layer[i,ix]
                iy1 = self.layer[i+1,ix]
                self.velocity[ix,iy0:iy1] = velseed[i]
        self.filled = True

    def fill_vlin_velocity(self,velseed):
        for ix in range(self.nx):
            for i in range(self.nlayers):
                iy0 = self.layer[i,ix]
                iy1 = self.layer[i+1,ix]
                self.velocity[ix,iy0:iy1] = np.linspace(velseed[i,0],velseed[i,1],iy1-iy0)
        self.filled = True

    def fill_velocity(self, velseed):
        if self.veltype == 'constant':
            self.fill_const_velocity(velseed)
        elif self.veltype == 'vlinear' or self.veltype == 'linear':
            self.fill_vlin_velocity(velseed)

    def set_velocity(self, vel):
        if self.shape == vel.shape:
            self.velocity = vel
            self.filled = True
        else:
            errexit("Wrong velocity shape: expected %s, got %s"%(self.shape, vel.shape))

    def set_layer(self, layer):
        layer_shape = (self.nlayers+1, self.nx)
        if layer.shape == layer_shape:
            self.layer = layer
        else:
            errexit("Wrong layer shape: expected %s, got %s"%(layer_shape, layer.shape))

    def add_history(self, key, val):
        self.history[key].append(val)

    def get_history(self, key, idx=-1):
        return self.history[key][idx]

    def clear_history(self):
        self.history=defaultdict(list)
        self.filled=False

    def generate(self,force_fill=False):
        if force_fill:
            self.filled=False
        if not self.filled:
            velseed = self.random.perturb(self.velseed, self.max_pert, fix_top=True)
            self.fill_velocity(velseed)
        return self.velocity


class Pipeline:
    def __init__(self, steps, model=None):
        self.steps = steps
        self.model = model

    def generate(self, model=None, clear=True):
        m = model or self.model
        if m is None:
            errexit("A model is required")
        if clear:
            m.clear_history()
        for step in self.steps:
            m = step.generate(m)
        return m.generate()


