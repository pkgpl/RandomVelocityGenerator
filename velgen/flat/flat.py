import numpy as np


class LayerDepthGenerator:
    def __init__(self, nlayers, ny, random_seed=None,
            top=0.2, bottom=0.9, minsplit=0.08):
        self.rng = np.random.default_rng(random_seed)
        self.nlayers = nlayers
        self.ny = ny
        # interface
        self.intf_nround=2
        self.top=top
        self.bottom=bottom
        self.minsplit=minsplit
        
    def _base_layer_depth(self):
        layer_depth = np.sort(self.top + self.rng.uniform(low=0,high=self.bottom-self.top,size=self.nlayers-1))
        layer_depth = np.append(np.insert(layer_depth,0,0),1)
        return np.round(layer_depth, self.intf_nround)
    
    def generate(self):
        layer_frac = self._base_layer_depth()
        diff = layer_frac[1:]-layer_frac[:-1]
        while diff.min() < self.minsplit:
            layer_frac = self._base_layer_depth()
            diff = layer_frac[1:]-layer_frac[:-1]
        self.layer_depth = layer_frac * self.ny
        return self.layer_depth


class VelocityGenerator:
    def __init__(self, shape, velseed, random_seed=None, fix_vtop=True, velmaxpert=0.1):
        self.rng = np.random.default_rng(random_seed)
        self.shape = shape
        self.nx, self.ny = self.shape
        self.nlayers = len(velseed)
        # velocity
        self.velseed=velseed
        self.vel_nround=2
        self.velpert=velmaxpert
        self.fix_vtop=fix_vtop

    def _gen_vels(self):
        vels = np.array(self.velseed)
        vels += self.rng.uniform(low=-1,high=1,size=self.nlayers)*self.velpert
        self.vels = np.round(vels, self.vel_nround)
        if self.fix_vtop:
            self.vels[0] = self.velseed[0]

    def _gen_layer_depth(self):
        layer_depth_generator = LayerDepthGenerator(self.nlayers, self.ny)
        self.layer_depth = layer_depth_generator.generate()
        self.ninterface = len(self.layer_depth)
        
    def _gen_layers(self):
        layers = np.zeros((self.ninterface,self.nx),dtype=np.int32)
        layers[-1,:]=self.ny
        for i in range(1,self.ninterface-1):
            layers[i,:] = self.layer_depth[i]
        self.layers = layers
        
    def set_layers(self,layers):
        self.layers = layers
    
    def fill_velocity(self):
        vel=np.zeros(self.shape,dtype=np.float32)
        for ix in range(self.nx):
            for i in range(1,self.ninterface):
                iy0=self.layers[i-1,ix]
                iy1=self.layers[i,ix]
                vel[ix,iy0:iy1] = self.vels[i-1]
        return vel
    
    def velocity(self):
        self._gen_vels()
        self._gen_layer_depth()
        self._gen_layers()
        return self.fill_velocity()


