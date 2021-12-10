import numpy as np
from .model import Random


class GaussianSalt:
    def __init__(self, vsalt=4.5, x0=None, height=None, width=None, eff_space=0.02, minspace=0.01,
            x0_range=(0.2,0.8), height_range=(0.2,0.6), width_range=(0.1,0.2),
            penetrate_interface=None, random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.vsalt=vsalt

        self.x0 = x0
        self.height = height
        self.width = width
        self.eff_space = eff_space
        self.minspace = minspace

        self.x0_range = x0_range
        self.height_range = height_range
        self.width_range = width_range
        self.penetrate_interface = penetrate_interface
        self.downshift = 0.02

    def _salt(self, A, sigma, x0, nx, ny):
        x = np.arange(nx)
        d1 = 2 * sigma**2
        return A * np.exp(-(x-x0)**2/d1) - self.downshift * ny

    def _add_salt_to_interface(self,interface,nx,ny):
        interface_with_salt=interface.copy()
        ninterface = len(interface)
        neff_space = self.eff_space * ny
        iminspace = self.minspace * ny

        x0     = self.x0     or self.random.uniform(*self.x0_range)*nx
        height = self.height or self.random.uniform(*self.height_range)*ny
        width  = self.width  or self.random.uniform(*self.width_range)*nx

        s = self._salt(height, width, x0, nx,ny)
        salt_top = ny - s
        self.salt_top = salt_top

        if self.penetrate_interface is None:
            penetrate_interface = self.random.choice([True,False])
        else:
            penetrate_interface = self.penetrate_interface
        if penetrate_interface:
            for i in range(1,ninterface-1):
                hdiff = salt_top[:] - interface[i,:]
                if hdiff.min() < neff_space:
                    height = abs(hdiff.min()) * (i/ninterface)
                    scale = 1 + (ninterface - i)/ninterface
                    swidth = width * scale
                    si = self._salt(height,swidth,x0,nx,ny)
                    interface_with_salt[i] -= si.astype(np.int32)
        else:
            for i in range(1,ninterface-1):
                hdiff = salt_top[:] - interface[i,:]
                if hdiff.min() < neff_space:
                    height = abs(hdiff.min()) + iminspace
                    scale = 1 + (ninterface - i)/ninterface
                    swidth = width * scale
                    si = self._salt(height,swidth,x0,nx,ny)
                    interface_with_salt[i] -= si.astype(np.int32)
            interface_with_salt = self._adjust_interface(interface_with_salt,iminspace)

        return interface_with_salt
        
    def _adjust_interface(self,interface,iminspace):
        ninterface = len(interface)
        for i in reversed(range(1,ninterface)):
            hdiff = interface[i]-interface[i-1]
            dmin = hdiff.min()
            if dmin < iminspace:
                interface[i-1,:] = interface[i-1,:] - abs(dmin) - int(iminspace)
        return interface

    def _add_salt_to_velocity(self,vel,salt_top,vsalt):
        nx = len(salt_top)
        for ix in range(nx):
            vel[ix,int(salt_top[ix]):] = vsalt
        return vel

    def generate(self, model):
        nx,ny = model.shape
        # add salt
        model.set_interface(self._add_salt_to_interface(model.interface,nx,ny))
        vel = model.generate(force_fill=True)
        model.add_history('gaussian_salt_top',self.salt_top)
        model.add_history('gaussian_vsalt',self.vsalt)
        salt_tops = model.history['gaussian_salt_top']
        vsalts = model.history['gaussian_vsalt']
        for salt_top,vsalt in zip(salt_tops, vsalts):
            model.set_velocity(self._add_salt_to_velocity(vel,salt_top,vsalt))
        return model



class EllipticSalt:
    def __init__(self, vsalt=4.5, center=None, a=None, b=None,
            x0_range=(0.1,0.9), y0_range=(0.4,0.9),
            width_range=(0.3,0.5), height_range=(0.08,0.2),
            vwidth_range=(0.1,0.2), vheight_range=(0.25,0.5),
            vertical=None,
            random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.vsalt=vsalt
        self.center=center
        self.a=a
        self.b=b
        self.x0_range=x0_range
        self.y0_range=y0_range
        self.a_range=np.array(width_range)/2
        self.b_range=np.array(height_range)/2
        self.va_range=np.array(vwidth_range)/2
        self.vb_range=np.array(vheight_range)/2
        self.vertical = vertical

    def mask(self,nx,ny,x0,y0,a,b):
        ax = np.arange(nx)
        ay = np.arange(ny)
        yy,xx = np.meshgrid(ay,ax)
        mask = ((xx-x0)/a)**2 + ((yy-y0)/b)**2 <= 1
        return mask

    def generate(self, model):
        nx,ny=model.shape
        vertical = self.vertical

        if self.center is not None:
            x0,y0 = self.center
        else:
            if 'gaussian_salt_top' in model.history:
                salt_top = model.get_history('gaussian_salt_top')
                x0 = np.argmin(salt_top)
                y0 = np.min(salt_top)
                vertical = False
            else:
                x0 = self.random.uniform(*self.x0_range)*nx
                y0 = self.random.uniform(*self.y0_range)*ny

        if vertical is None:
            vertical = self.random.choice((True,False))
        if vertical:
            a_range, b_range = self.va_range, self.vb_range
        else:
            a_range, b_range = self.a_range, self.b_range
        a= self.a or self.random.uniform(*a_range)*nx
        b= self.b or self.random.uniform(*b_range)*ny

        # salt does not penetrate the first interface
        if y0 - b < model.interface[1].min():
            y0 += model.interface[1].min()

        mask = self.mask(nx,ny,x0,y0,a,b)
        vel = model.generate()
        vel[mask] = self.vsalt
        model.set_velocity(vel)
        model.add_history('elliptic_center',(x0,y0))
        model.add_history('elliptic_ab',(a,b))
        model.add_history('elliptic_vsalt',self.vsalt)
        return model
