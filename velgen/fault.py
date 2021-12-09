import numpy as np
from .model import Random


class LinearFault:
    def __init__(self, nfaults=None, vshift=None, max_nfaults=3, lpad=0.1, rpad=0.1, vshift_range=(0.05,0.15), random_seed=None, vround=4):
        self.random = Random(random_seed, vround)
        self.nfaults = nfaults
        self.vshift = vshift
        self.max_nfaults = max_nfaults
        self.lpad=lpad
        self.rpad=rpad
        self.vshift_min=vshift_range[0]
        self.vshift_max=vshift_range[1]

    def _gen_fault_lines(self,nfaults,nx):
        top = self.random.array(low=self.lpad, high=1-self.rpad, size=nfaults)
        bottom = self.random.array(low=self.lpad, high=1-self.rpad, size=nfaults)
        itop = (top*nx).astype(np.int32)
        ibottom = (bottom*nx).astype(np.int32)
        return itop, ibottom

    def _get_vshifts(self,nfaults,ny):
        ivshift_min=int(ny * self.vshift_min)
        ivshift_max=int(ny * self.vshift_max)
        if self.vshift is None:
            shift_sign = self.random.choice((-1,1))
            vshifts = self.random.uniform(ivshift_min, ivshift_max, size=nfaults, dtype=np.int32)*shift_sign
        else:
            vshifts = np.ones(nfaults, dtype=np.int32) * self.vshift
        return vshifts

    def add_fault(self, vel, it, ib, vshift, ny):
        fault_line = np.linspace(it,ib,ny).astype(np.int32)
        igrad = abs(it-ib)/ny

        hshift = int(igrad*vshift)
        vpad = abs(vshift)
        hpad = abs(hshift)

        velpad = np.pad(vel, [(hpad,hpad),(vpad,vpad)],mode='edge')
        velnew = vel.copy()

        for iy in range(ny):
            for ix in range(fault_line[iy]):
                velnew[ix,iy] = velpad[ix+hpad-hshift, iy+vpad-vshift]
        return velnew

    def generate(self, model):
        nx,ny = model.shape
        if self.nfaults is not None:
            nfaults = self.nfaults 
        else:
            nfaults = self.random.choice(np.arange(1,self.max_nfaults+1,dtype=np.int32))
        itops,ibottoms = self._gen_fault_lines(nfaults,nx)
        vshifts = self._get_vshifts(nfaults,ny)

        vel = model.generate()
        for i,(it,ib) in enumerate(zip(reversed(itops), reversed(ibottoms))):
            vel = self.add_fault(vel,it,ib,vshifts[i],ny)
        model.set_velocity(vel)
        model.add_history('fault_top',itops)
        model.add_history('fault_bottom',ibottoms)
        return model


