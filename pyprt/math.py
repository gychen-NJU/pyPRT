from tracemalloc import stop

from .needs import *
import warnings
warnings.filterwarnings('ignore', message='Using a non-tuple sequence for multidimensional indexing is deprecated')

def to_tensor(data, **kwargs):
    device = kwargs.get('device',None)
    dtype  = kwargs.get('dtype' ,None)
    if isinstance(data, np.ndarray):
        ret = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        ret = data
    elif isinstance(data, float, list, tuple):
        ret = torch.tensor(data)
    else:
        raise TypeError(f"Unsupported types: {type(data)}")

    if device is not None:
        ret = ret.to(device)
    if dtype is not None:
        ret = ret.to(dtype)
    return ret

def VoigtFaradayFunction(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=50,
    lim=10.0,
) -> torch.Tensor:
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim,lim,ynodes,device=device,dtype=dtype)
    dy = y[1]-y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None,None,:]
    numerator = torch.exp(-y**2)*(u-y)
    denominator = (u-y)**2+a**2
    integrand = numerator/denominator
    profile = (1/torch.pi**1.5)*torch.trapz(integrand,dx=dy,dim=-1)
    return profile

def VoigtFunction(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=50,
    lim=10.0,
) -> torch.Tensor:
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim,lim,ynodes,device=device,dtype=dtype)
    dy = y[1]-y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None,None,:]
    try:
        numerator = torch.exp(-y**2)
        denominator = (u-y)**2+a**2
        integrand = numerator/denominator
        profile = (a[:,:,0]/torch.pi**1.5)*torch.trapz(integrand,dx=dy,dim=-1)
    except:
        print(u.shape,a.shape,y.shape)
        raise
    return profile

class TensorGridInterpolator:
    '''
    Irregular Grid Interpolator
    '''
    def __init__(self, grids, values, method='bilinear', bounds_error=True,fill_value=None):
        """
        Initialize the interpolator

        Parameters:
        -----------
        grids: tuple
            ( xgrid, ygrid (,zgrid) )
        values: 2D/3D tensor
            values in the grid points, must match the grid's dimension
        method: str (default -- 'bilinear')
            interpolation method ('bilinear', 'nearest')
        bounds_error: bool (default -- True)
            whether reporting an error when boundary is exceeded
        fill_value: float (default -- None)
            filling value when exceeding the boundary
        """
        self.grids  = grids
        self.values = values
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.dims = len(grids)
        self.device = values.device
        self.dtype = values.dtype

        if self.dims!=values.dim():
            raise ValueError(f"grid's dimension: {self.dims} does not match the `values`'s: {values.dim()}")

        self.grid_info = []
        for i,grid in enumerate(grids):
            step = torch.diff(grid)
            is_uniform = torch.allclose(step,step[0])
            grid_min = grid[0]
            grid_max = grid[-1]
            self.grid_info.append(
                dict(
                    grid=grid,
                    step=step,
                    is_uniform=is_uniform,
                    lower = grid_min,
                    upper = grid_max,
                    size = len(grid)
                ))
        while self.values.dim()<self.dims+2:
            self.values = self.values.unsqueeze(0) # -> [1,1,H,W,D]

    def to(self,device=None,dtype=None):
        flag1 = (device is not None and device!=self.device)
        flag2 = (dtype is not None and dtype!=self.dtype)
        if flag1 or flag2:
            grids_new = []
            for igrid in grids:
                grids_new.append(igrid.to(device=device,dtype=dtype))
            grids_new = tuple(grids_new)
            values_new = values.to(device=device,dtype=dtype)
            self.__init__(grids_new,values_new,self.method,self.bounds_error,self.fill_value)

    def __call__(self, points):
        """
        Interpolating at the given points

        Parameter:
        ----------
        points: tensor (shape: (...,2 or 3))
            the coordinates for the interpolating points

        Return:
        ---------
        result: tensor
            interpolation results
        """
        original_dim = points.dim()
        while points.dim()<self.dims+2:
            points = points.unsqueeze(0) # -> [1, H0, W0, (D0,) dims]
        if points.shape[-1] != self.dims:
            raise ValueError(f"Coordinates dimension should be (...,{self.dims}), but get {points.shape}")
        grid_coords = []
        for idim in range(self.dims):
            xi = points[...,idim] # coordinates in dimension i
            xi = self._check_bounds(xi, dim=idim)
            ix = self._find_indices(xi, dim=idim) # indicies
            xn = self._normalize(xi,ix, dim=idim) # normalized to [-1,1]
            grid_coords.append(xn)
        grid_coords = torch.stack(grid_coords,dim=-1)
        values = self.values
        result = F.grid_sample(
            values,
            grid_coords,
            mode=self.method,
            padding_mode = 'zeros' if self.fill_value is None else 'border',
            align_corners = True
        ) # -> [1, H0, W0(, D0),1]
        result = result.squeeze(0)
        while result.dim()>original_dim-1:
            result = result.squeeze(0)
        return result

    def _find_indices(self, xi, dim=0):
        igrid = self.grid_info[dim]['grid']
        if self.grid_info[dim]['is_uniform']:
            return None
        else:
            indices = torch.searchsorted(igrid, xi.contiguous())
            indices = torch.clamp(indices, 1, len(igrid)-1)
            return indices

    def _normalize(self, xi, indices, dim=0):
        igrid = self.grid_info[dim]['grid']
        vmin  = self.grid_info[dim]['lower']
        vmax  = self.grid_info[dim]['upper']
            
        if self.grid_info[dim]['is_uniform']:
            xn = 2.0*(xi-vmin)/(vmax-vmin)-1.0
        else:
            lower = igrid[indices-1]
            upper = igrid[indices]
            normalized = (xi-lower)/(upper-lower)
            lgrid = 2.0*(indices-1)/(len(igrid)-1)-1.0
            ugrid = 2.0*indices/(len(igrid)-1)-1.0
            xn = lgrid+normalized*(ugrid-lgrid)
        return xn

    def _check_bounds(self, xi, dim=0):
        below = xi<self.grid_info[dim]['lower']
        above = xi>self.grid_info[dim]['upper']
        out_of_bounds = below | above

        if torch.any(out_of_bounds):
            if self.bounds_error:
                raise ValueError(
                    f"dimension-{dim} out of bounds: {xi[out_of_bounds]}"
                )
            else:
                if self.fill_value is not None:
                    xi[out_of_bounds] = self.fill_value
                else:
                    xi[below] = self.grid_info[dim]['lower']
                    xi[above] = self.grid_info[dim]['upper']
        return xi

class RK4_solver():
    def __init__(self, func):
        self.func = func

    def stepper(self, x0, y0, dx):
        k1 = dx*self.func(x0       ,y0)
        k2 = dx*self.func(x0+0.5*dx,y0+0.5*k1)
        k3 = dx*self.func(x0+0.5*dx,y0+0.5*k2)
        k4 = dx*self.func(x0+1.0*dx,y0+1.0*k3)
        return y0+1/6*(k1+2*k2+2*k3+k4)

    def __call__(self, xlist, y0):
        ysol = [y0]
        yi = y0
        for i,xi in enumerate(xlist[:-1]):
            dx = xlist[i+1]-xlist[i]
            yi = self.stepper(xi,yi,dx)
            ysol.append(yi)
        return ysol
            
class Newton_solver():
    def __init__(self, func):
        self.func = func

    def stepper(self, x0, x1):
        eps = 1e-8
        denominator = self.func(x1)-self.func(x0)
        denominator = torch.where(torch.abs(denominator)<=eps,eps*torch.sign(denominator),denominator)
        x2 = x1-self.func(x1)*(x1-x0)/denominator
        return x2

    def __call__(self,x0,x1,nsteps=10):
        for istep in range(nsteps):
            if torch.allclose(x0,x1, atol=1e-2):
                return x1
            else:
                x2 = self.stepper(x0,x1)
                x0 = x1.clone()
                x1 = x2.clone()
        return x1

def derivative(y, x, dim=1):
    y_dim = y.dim()
    x_dim = x.dim()
    if x_dim < y_dim:
        num_new_dims = y_dim - x_dim
        x = x[(...,) + (None,) * num_new_dims]
        x = x.expand_as(y)
    
    dy = torch.zeros_like(y)
    idx = [slice(None)] * y_dim
    idxi = idx[:dim]+[0]+idx[dim+1:]
    idxj = idx[:dim]+[1]+idx[dim+1:]
    dy[0] = (y[idxj]-y[idxi])/(x[idxj]-x[idxi])
    idxi = idx[:dim]+[-2]+idx[dim+1:]
    idxj = idx[:dim]+[-1]+idx[dim+1:]
    dy[-1] = (y[idxj]-y[idxi])/(x[idxj]-x[idxi])
    idxh = idx[:dim]+[slice(0,-2)]+idx[dim+1:]
    idxi = idx[:dim]+[slice(1,-1)]+idx[dim+1:]
    idxj = idx[:dim]+[slice(2,None)]+idx[dim+1:]
    d1 = 1/(x[idxi]-x[idxj])
    d2 = 1/(x[idxh]-x[idxi])
    d3 = 1/(x[idxh]-x[idxj])
    dy[idxi] = y[idxj]*(d3-d1)+y[idxi]*(d1-d2)+y[idxh]*(d2-d3)
    return dy

rte_dict = {}

def register_rte(key):
    def decorator(func):
        rte_dict[key] = func
        return func
    return decorator

@register_rte('RK4')
def RK4_method(ltau, K_matx, S_func):
    x  = ltau.squeeze()
    K  = K_matx*np.log(10)*torch.pow(10,ltau[:,:,:,None,None])
    S  = S_func.unsqueeze(-1)
    Il = S[:,-1].clone()
    # Il = torch.zeros_like(S[:,-1])
    def rfunc(xi, Ii):
        idx = torch.searchsorted(x,xi)
        Ki  = K[:,idx]*(xi-x[idx-1])/(x[idx]-x[idx-1])+K[:,idx-1]*(x[idx]-xi)/(x[idx]-x[idx-1])
        Si  = S[:,idx]*(xi-x[idx-1])/(x[idx]-x[idx-1])+S[:,idx-1]*(x[idx]-xi)/(x[idx]-x[idx-1])
        # Ki = K[:,idx]
        # Si = S[:,idx]
        return torch.matmul(Ki, Ii-Si)
    stepper = RK4_solver(rfunc).stepper
    dx = torch.diff(x.flip(0))
    for xl,dxl in zip(x.flip(0)[:-1],dx):
        Ik = stepper(xl, Il, dxl)
        Il = Ik
    return Ik.squeeze(-1)

@register_rte('Hermitian')
def Hermitian_method(ltau, K_matx, S_func):
    """
    Bellot Rubio, L. R., Ruiz Cobo, B., & Collados, M. 1998, ApJ, 506, 805
    Hermition Algortithm to calculation the polarization radiative transfer equation
    """
    Nb,Nt,Nw = K_matx.shape[:3]
    K_matx = K_matx.flip(1)
    S = S_func.flip(1).unsqueeze(-1)
    ltau = ltau.flip(1)[...,None,None]
    tau = torch.pow(10,ltau) # (Nb,Nt,Nw,1,1)
    dtau = tau[:,:-1]-tau[:,1:]
    # print(f"K: {K_matx.shape} | tau: {tau.shape}")
    dzi = 0.5*(ltau[:,1:]-ltau[:,:-1])
    d2i = dzi**2/3
    # print(K_matx.shape,tau.shape)
    K = K_matx*np.log(10)*tau
    try:
        Kinv = torch.linalg.inv(K)
    except RuntimeError as e:
        Kinv = torch.linalg.pinv(K)
    dS = derivative(S, ltau, dim=1)
    Ii = torch.matmul(Kinv[:,0],dS[:,0])+S[:,0]
    K2 = torch.matmul(K,K)
    dK = derivative(K, ltau, dim=1)
    eye = torch.eye(4)[None,None,:,:].to(device=ltau.device,dtype=ltau.dtype)
    for j in range(1,Nt):
        dz = dzi[:,j-1]
        d2 = d2i[:,j-1]
        Ki = K[:,j-1]
        Kj = K[:,j]
        Si = S[:,j-1]
        Sj = S[:,j]
        dKi = dK[:,j-1]
        dKj = dK[:,j]
        dSi = dS[:,j-1]
        dSj = dS[:,j]
        K2i = K2[:,j-1]
        K2j = K2[:,j]
        # print(f"Ki: {Ki.shape} | dz: {dz.shape} | Si: {Si.shape} | Ii: {Ii.shape}")
        den = -(torch.matmul(Ki,Si)+torch.matmul(Kj,Sj))*dz+Ii
        den+= -(
            torch.matmul(dKi,Si)-torch.matmul(dKj,Sj)+
            torch.matmul(Ki,dSi)-torch.matmul(Kj,dSj)+
            torch.matmul(K2i,Si)-torch.matmul(K2j,Sj)
        )*d2
        den+= torch.matmul(Ki*dz+d2*(dKi+K2i),Ii)
        num = -Ki*dz+d2*(dKi+K2i)+eye
        try:
            num = torch.linalg.inv(num)
        except RuntimeError as e:
            num = torch.linalg.pinv(num)
        # print(f"num: {num.shape} | den: {den.shape} | Ki: {Ki.shape} | dKi: {dKi.shape} | Si: {Si.shape} | dSi: {dSi.shape}")
        stop
        Ij = torch.matmul(num,den)
        Ii = Ij.clone()
    # print(Ij.shape)
    return Ij.squeeze(-1)

@register_rte('DELO')
def DELO_method(ltau, K_matx, S_func):
    tau = torch.pow(10,ltau.squeeze())
    eye = torch.eye(4)[None,None,None,:,:]
    Kp  = K_matx-eye
    Sp  = torch.matmul(K_matx,S_func.unsqueeze(-1))
    dt  = torch.zeros_like(tau)
    dt[:-1]  = torch.diff(tau)
    dt[-1] = torch.diff(tau)[-1]
    E   = torch.exp(-dt)
    F   = 1-E
    G   = (1-(1+dt)*E)/dt
    Il  = S_func[:,-1].unsqueeze(-1)
    for l in range(len(tau)-1,0,-1):
        k  = l-1
        Ek = E[k]
        Fk = F[k]
        Gk = G[k]
        Sp_k = Sp[:,k]
        Kp_k = Kp[:,k]
        Sp_l = Sp[:,l]
        Kp_l = Kp[:,l]
        temp = eye[0]+(Fk-Gk)*Kp_k
        try:
            inv = torch.linalg.inv(temp)
        except RuntimeError as e:
            inv = torch.linalg.pinv(temp)
        Pscr_k = torch.matmul(inv, (Fk-Gk)*Sp_k+Gk*Sp_l)
        Lscr_k = torch.matmul(inv, Ek*eye[0]-Gk*Kp_l)
        Ik = Pscr_k+torch.matmul(Lscr_k,Il)
        Il = Ik.clone()
    # print(Ik.shape)
    return Ik.squeeze(-1)

def RTE_solver(solver_name):
    if solver := rte_dict.get(solver_name):
        return solver
    else:
        raise ValueError(f"`{solver_name}` is not supported. Please choose from {list(rte_dict.keys())}")

def polynomial_partition_function(polyQ, T):
    lnQ = torch.sum(
        torch.stack(
            [ai*torch.log(T)**i for i,ai in enumerate(polyQ)],
            dim=0),
        dim=0
    )
    Q = torch.exp(lnQ)
    return Q

def signclamp(x, min=None, max=None):
    ret = torch.where(x>=0, torch.clamp(x, min=min, max=max), -torch.clamp(-x,min=min,max=max))
    return ret

def f1_formal_sol1(g1,g2,g3,g4,g5):
    # derived by Walfram Mathematica
    ret = torch.sqrt((g2-g3)**2+g1**2*(1+g2+g3)**2+4*g2*g4+2*g1*(-(g3*(1+g3))+g2*(1+g2+4*g4)+4*g5))
    ret = ret* (g2*g4 + g5)
    ret = ret*torch.sign(g2*(g2**2-(1+g3)*(1+3*g3)-2*g2*(g3-2*g4))*g4+2*(g2+g2**2-g3*(1+g3)+4*g2*g4)*g5+4*g5**2)
    ret = ret+g2*(1 + 2*g3 + g1*(1 + g2 + g3))*g4 + (-g2 + g3 + g1*(1 + g2 + g3))*g5
    ret = ret/(g2*(g2**2-(1+g3)*(1+3*g3)-2*g2*(g3-2*g4))*g4+2*(g2+g2**2-g3*(1+g3)+4*g2*g4)*g5+4*g5**2)
    ret = -ret
    return ret

def f1_formal_sol2(g1,g2,g3,g4,g5):
    ret = torch.sqrt((g2-g3)**2+g1**2*(1+g2+g3)**2+4*g2*g4+2*g1*(-(g3*(1+g3))+g2*(1+g2+4*g4)+4*g5))
    ret = ret* (g2*g4 + g5)
    ret = ret*torch.sign(g2*(g2**2-(1+g3)*(1+3*g3)-2*g2*(g3-2*g4))*g4+2*(g2+g2**2-g3*(1+g3)+4*g2*g4)*g5+4*g5**2)
    ret = ret-(g2*(1 + 2*g3 + g1*(1 + g2 + g3))*g4) - (-g2 + g3 + g1*(1 + g2 + g3))*g5
    ret = ret/(g2*(g2**2-(1+g3)*(1+3*g3)-2*g2*(g3-2*g4))*g4+2*(g2+g2**2-g3*(1+g3)+4*g2*g4)*g5+4*g5**2)
    return ret

def lagrange_interp(x,yh,yi,yj,xh,xi,xj):
    """
    Lagrange interpolation
    """
    d1 = x-xh
    d2 = x-xi
    d3 = x-xj
    d12 = xh-xi
    d13 = xh-xj
    d23 = xi-xj
    ret = yh*d2*d3/(d12*d13) + yi*d1*d3/(d23*d13) + yj*d1*d2/(d12*d23)
    return ret


def voigt2(v,a):
    """
    Compute the Voigt and VoigtFaraday profile.
    
    Parameters
    ----------
    v : torch.Tensor # [Nb,Nt,Nw]
        The input tensor.
    a : torch.Tensor # [Nb,Nt,Nw]
        The input tensor.
        
    Returns
    -------
    H : torch.Tensor
        The Voigt profile.
    F : torch.Tensor
        The VoigtFaraday profile.
    """
    A0,A1,A2,A3,A4,A5,A6,B0,B1,B2,B3,B4,B5,B6 = (
        122.607931777104326,214.382388694706425,181.928533092181549,  93.155580458138441,30.180142196210589,
        5.912626209773153,  .564189583562615,122.60793177387535,352.730625110963558,  457.334478783897737,
        348.703917719495792,170.354001821091472,  53.992906912940207,10.479857114260399
    )
    xdws = torch.tensor([
        .1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.2,1.4,1.6,1.8,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.
    ],device=v.device,dtype=v.dtype)
    ydws = torch.tensor([
        9.9335991E-02,1.9475104E-01,2.8263167E-01,3.5994348E-01,  
        4.2443639E-01,4.7476321E-01,5.1050407E-01,5.3210169E-01,  
        5.4072434E-01,5.3807950E-01,5.0727350E-01,4.5650724E-01,  
        3.9993989E-01,3.4677279E-01,3.0134040E-01,1.7827103E-01,  
        1.2934799E-01,1.0213407E-01,8.4542692E-02,7.2180972E-02,  
        6.3000202E-02,5.5905048E-02,5.0253846E-02,4.1812878E-02,  
        3.5806101E-02,3.1311397E-02,2.7820844E-02,2.5031367E-02
    ],device=v.device,dtype=v.dtype)
    ivsigno = torch.where(v<0,-1,1)
    v = ivsigno*v
    a_eq_zero = torch.isclose(a,torch.zeros_like(a),atol=1e-6)
    if torch.any(a_eq_zero):
        v2 = v**2
        H  = torch.exp(-v2)
        k  = torch.searchsorted(xdws,v, right=False).clamp(min=1,max=len(xdws)-2)
        D  = lagrange_interp(v,ydws[k-1],ydws[k],ydws[k+1],xdws[k-1],xdws[k],xdws[k+1])
        D  = torch.where(v<=xdws[0],v*(1-2/3*v2),D)
        D  = torch.where(v>=xdws[-1],.5/v*(1+.5/v2),D)
        F  = ivsigno*5.641895836E-1*D
        if torch.all(a_eq_zero):
            return H,F
    z = torch.complex(a,-v)
    z = ((((((A6*z+A5)*z+A4)*z+A3)*z+A2)*z+A1)*z+A0)/(((((((z+B6)*z+B5)*z+B4)*z+B3)*z+B2)*z+B1)*z+B0)
    if torch.any(a_eq_zero):
        H = torch.where(a_eq_zero,H,z.real)
        F = torch.where(a_eq_zero,F,z.imag*ivsigno*.5)
    else:
        H = z.real
        F = z.imag*ivsigno*.5
    return H,F