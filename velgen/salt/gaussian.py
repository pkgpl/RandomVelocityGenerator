import numpy as np


class GaussianSaltGenerator:
    def __init__(self,velgen,x0=None,height=None,width=None,eff_space=0.05,minspace=0.04,vel_salt=4.5,random_seed=None):
        self.rng = np.random.default_rng(random_seed)
        self.velgen = velgen
        self.nx,self.ny = velgen.nx,velgen.ny
        # salt
        self.vel_salt = vel_salt
        self.x = np.arange(self.nx)

        self.x0     = x0     * self.nx if x0     else None
        self.height = height * self.ny if height else None
        self.width  = width  * self.nx if width  else None

        self.eff_space = eff_space
        self.neff_space = eff_space * self.ny
        self.minspace = minspace * self.ny

        self.x0_min = 0.2
        self.x0_max = 0.8
        self.height_min = 0.2
        self.height_max = 0.6
        self.width_min = 0.1
        self.width_max = 0.15
        self.salt_down = 0.01 * self.ny

    def _iminmax(self,valmin,valmax, n):
        return self.rng.uniform(valmin, valmax) * n

    def _salt(self,A,sigma,x0):
        x = self.x
        
        d1 = 2 * sigma**2
        return A*np.exp(-(x-x0)**2/d1) - self.salt_down

    def _add_salt_to_layers(self):
        layers_with_salt=self.layers.copy()

        x0     = self.x0     if self.x0     else self._iminmax(self.x0_min,     self.x0_max,     self.nx)
        height = self.height if self.height else self._iminmax(self.height_min, self.height_max, self.ny)
        width  = self.width  if self.width  else self._iminmax(self.width_min,  self.width_max , self.nx)

        s = self._salt(height, width, x0)
        salt_top = self.ny - s

        for i in range(1,self.ninterface-1):
            hdiff = salt_top[:] - self.layers[i,:]
            if hdiff.min() < self.neff_space:
                height = abs(hdiff.min())+self.minspace
                scale = 1 + (self.ninterface - i)/self.ninterface
                width = width * scale
                si = self._salt(height,width,x0)
                layers_with_salt[i] -= (si).astype(np.int32)
        self.layers_with_salt = self._adjust_layers(layers_with_salt)
        self.salt_top = salt_top
        
    def _adjust_layers(self,layers):
        for i in reversed(range(1,self.ninterface)):
            hdiff = layers[i]-layers[i-1]
            dmin = hdiff.min()
            if dmin < self.minspace:
                layers[i-1,:] = layers[i-1,:] - abs(dmin) - int(self.minspace)
        return layers

    def _add_salt_to_velocity(self,vel):
        for ix in range(self.nx):
            vel[ix,int(self.salt_top[ix]):] = self.vel_salt
        return vel
    
    def velocity(self):
        # generate velocity
        self.velgen.velocity()
        self.layers = self.velgen.layers
        self.ninterface = len(self.layers)
        # add salt
        self._add_salt_to_layers()
        self.velgen.set_layers(self.layers_with_salt)
        vel = self.velgen.fill_velocity()
        return self._add_salt_to_velocity(vel)

