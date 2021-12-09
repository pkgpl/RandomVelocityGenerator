import numpy as np
from .model import Random


class FlatLayer:
    def __init__(self, y_range=(0.1,0.9),minsplit=0.05,depth=None, random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.vround=vround
        self.minsplit = minsplit
        self.y_range = np.array(y_range)
        self.depth = depth

    def generate(self, model):
        nx, ny = model.shape
        ninterf = model.nlayers - 1
        iminsplit = int(ny * self.minsplit)
        iy_range = (ny * self.y_range).astype(np.int32)

        depth = self.depth or self.random.array_interval(*iy_range,ninterf,iminsplit, prepend=0,append=ny)
                
        layer = np.repeat(depth, nx)
        model.set_layer(np.reshape(layer, (len(depth), nx)).astype(np.int32))
        model.add_history('flat_depth',depth)
        return model


class DippingLayer:
    def __init__(self, y_range=(0.1,0.9),minsplit=0.05, left=None, right=None,random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.vround=vround
        self.minsplit = minsplit
        self.y_range = np.array(y_range)
        self.left = left
        self.right = right

    def generate(self, model):
        nx, ny = model.shape
        ninterf = model.nlayers - 1
        iminsplit = int(ny * self.minsplit)
        iy_range = (ny * self.y_range).astype(np.int32)

        left  = self.left  or self.random.array_interval(*iy_range,ninterf,iminsplit, prepend=0,append=ny)
        right = self.right or self.random.array_interval(*iy_range,ninterf,iminsplit, prepend=0,append=ny)

        nb = len(left)
        layer = np.zeros((nb, nx),dtype=np.int32)
        for i in range(nb):
            layer[i,:] = np.linspace(left[i],right[i],nx)
        model.set_layer(layer.astype(np.int32))
        model.add_history('dip_left', left)
        model.add_history('dip_right',right)
        return model


class LinearWaterLayer:
    def __init__(self, y_range=(0.1,0.4),vwater=1.5, left=None, right=None, random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.vround=vround
        self.y_range = np.array(y_range)
        self.vwater = vwater
        self.left = left
        self.right = right

    def generate(self, model):
        nx, ny = model.shape
        iy_range = (ny * self.y_range)

        left  = self.left or self.random.uniform(*iy_range)
        right = self.right or self.random.uniform(*iy_range)
        waterbottom = np.linspace(left,right,nx,dtype=np.int32)

        vel = model.generate()
        for ix in range(nx):
            vel[ix,:waterbottom[ix]] = self.vwater
        model.set_velocity(vel)
        model.add_history('water_bottom',waterbottom)
        return model


