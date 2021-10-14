import numpy as np


class CosineFoldGenerator:
    def __init__(self, flat, Amax=5, Hmax=5, random_seed=None):
        self.rng = np.random.default_rng(random_seed)
        self.flat = flat
        self.set_layers=self.flat.set_layers
        self.fill_velocity=self.flat.fill_velocity
        self.nx, self.ny = self.flat.shape
        # Fold
        self.Amax=Amax
        self.Hmax=Hmax
        
    def _gen_folded_layers(self):
        layers = np.zeros((self.ninterface,self.nx),dtype=np.int32)
        layers[-1,:]=self.ny
        fold = self._base_folded_layer()
        for i in range(1,self.ninterface-1):
            layers[i,:] = self.layer_depth[i] + fold[:]
        self.layers = layers
    
    def _base_folded_layer(self):
        r1 = np.sort(self.rng.random(self.nx))
        H1 = self.rng.uniform(low=1, high=self.Hmax)
        A1 = self.rng.uniform(low=1, high=self.Amax)
        return A1*np.cos(H1*np.pi*r1)
    
    def velocity(self):
        self.flat.velocity()
        self.layer_depth = self.flat.layer_depth
        self.ninterface = len(self.layer_depth)
        self._gen_folded_layers()
        self.set_layers(self.layers)
        return self.fill_velocity()


