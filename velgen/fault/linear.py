import numpy as np


class LinearFaultGenerator:
    def __init__(self, velgen, max_nfaults=2, lpad=0.1, rpad=0.1, vshift_min=0.05, vshift_max=0.15, random_seed=None):
        self.rng = np.random.default_rng(random_seed)
        self.velgen = velgen
        self.nx, self.ny = velgen.nx, velgen.ny
        self.max_nfaults = max_nfaults
        self.lpad=lpad
        self.rpad=rpad
        self.vshift_min=vshift_min
        self.vshift_max=vshift_max
        self.shift_sign = self.rng.choice((-1,1))
        
    def _gen_fault_lines(self):
        top = np.sort(self.rng.uniform(low=self.lpad,high=1-self.rpad,size=self.nfaults))
        bottom = np.sort(self.rng.uniform(low=self.lpad,high=1-self.rpad,size=self.nfaults))
        self.itop = (top*self.nx).astype(np.int32)
        self.ibottom = (bottom*self.nx).astype(np.int32)

    def _add_one_fault(self,vel,it,ib):
        fault_line = np.linspace(it,ib,self.ny).astype(np.int32)
        igrad = abs(it-ib)/self.ny
        
        vshift = int(self.rng.uniform(low=self.vshift_min,high=self.vshift_max)*self.ny*self.shift_sign)
        hshift = int(igrad*vshift)
        
        vpad = abs(vshift)
        hpad = abs(hshift)
        velpad = np.pad(vel,[(hpad,hpad),(vpad,vpad)],mode='edge')
        velnew = vel.copy()
        
        for iy in range(self.ny):
            for ix in range(fault_line[iy]):
                velnew[ix,iy] = velpad[ix+hpad-hshift,iy+vpad-vshift]
        return velnew

    def _add_faults(self,vel):
        for it,ib in zip(reversed(self.itop), reversed(self.ibottom)):
            vel = self._add_one_fault(vel,it,ib)
        return vel
    
    def velocity(self,nfaults=None):
        if nfaults is None:
            self.nfaults = self.rng.choice(range(1,self.max_nfaults+1))
        self._gen_fault_lines()
        vel = self.velgen.velocity()
        return self._add_faults(vel)


