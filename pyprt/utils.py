from typing import Any
from unittest import skip

from .needs import *
import pkg_resources

def lines_option(acquire=False):
    """
    Show the optional lines
    """
    lines_file = pkg_resources.resource_filename(
        'pyprt',
        'data/lines.json'
    )
    with open(lines_file) as f:
        lines = json.load(f)
    if acquire:
        return lines
    else:
        option = {iline:list[Any](lines[iline].keys()) for iline in lines.keys()}
        print(json.dumps(option, indent=2, ensure_ascii=False))

def load_lines(lines_usr):
    lines_sys = lines_option(acquire=True)
    lines = dict()
    for iline in lines_usr:
        if iline not in lines_sys.keys():
            raise KeyError(f"{iline} is not supported now. Please choose in {tuple(lines_sys.keys())}")
        else:
            for iwav in lines_usr[iline]:
                if iwav not in lines_sys[iline]:
                    raise KeyError(f"{iwav} is not supported now. Please choose in {tuple(lines_sys[iline].keys())}")
                else:
                    lines[f'{iline}_{iwav}'] = lines_sys[iline][iwav]
    return lines

def load_initial_guess(model='Umbral_big_spot'):
    path = pkg_resources.resource_filename(
        'pyprt',
        f'data/model_atmosphere'
    )
    files = sorted(glob.glob(os.path.join(path,"*.txt")))
    model_options = {os.path.basename(ifile).split(".")[0] for ifile in files}
    if model not in model_options:
        raise ValueError(f"{model} is not supported now. Please choose in {tuple(model_options)}")
    else:
        data = np.loadtxt(os.path.join(path,f"{model}.txt"), skiprows=2)
        initial_guess = dict(
            ltau        = data[:,0],
            T           = data[:,1],
            Pe          = data[:,2],
            vmic        = data[:,3],
            Bmag        = data[:,4],
            vLos        = data[:,5],
            inclination = data[:,6],
            azimuth     = data[:,7],
            Pg          = data[:,9],
            rhog        = data[:,10],
        )
        return initial_guess

def get_xyz(atomic_properties):
    """
    Calculate the atmospheric abundance X(H),Y(He),Z(metal)
    """
    amw = atomic_properties.amw # abu*wgt
    X = 1/amw
    Y = X*atomic_properties.abu[1]*atomic_properties.wgt[1]
    Z = 1-X-Y
    return X,Y,Z

def NormalizeNodes(ltau,nodes):
    device = nodes.device
    ltau = ltau.clone().squeeze().to(device)
    Nt = ltau.numel()
    idx = torch.searchsorted(ltau,nodes).clamp(min=1,max=Nt-1)
    lower = ltau[idx-1]
    upper = ltau[idx]
    normal = (nodes-lower)/(upper-lower)
    lgrid = 2.0*(idx-1)/(Nt-1)-1.0
    ugrid = 2.0*idx/(Nt-1)-1.0
    ltn = lgrid+normal*(ugrid-lgrid)
    # print('nodes : ',nodes)
    # print('normal: ',normal)
    # print('lower : ',lower)
    # print('upper : ',upper)
    # print('idx   : ',idx)
    # print('ltau  : ',ltau)
    return ltn

def grid2node(ltau,x,nnodes=1):
    Nt = ltau.numel()
    if nnodes >= Nt:
        return x
    Nb = x.size(0)
    tvbgf = x[:,:5*Nt].reshape(Nb,5,Nt)[:,:,None,:].clone() # (Nb,5,1,Nt)
    m = x[:,5*Nt  ]
    M = x[:,5*Nt+1]
    if nnodes==1:
        xnodes = tvbgf.mean(dim=-1,keepdim=True) # (Nb,5,1,1)
    elif nnodes==2:
        xnodes = torch.stack([tvbgf[:,:,:,0],tvbgf[:,:,:,-1]],dim=2) # (Nb,5,1,2)
    # elif nnodes==3:
    #     ltnodes = torch.zeros(Nb,5,nnodes,2).to(device=x.device,dtype=x.dtype)
    #     ltnodes[...,0] = torch.linspace(-1,1,3).to(x.device,dtype=x.dtype)[None,None,None,:]
    #     xnodes = F.grid_sample(
    #         tvbgf,
    #         ltnodes,
    #         mode='bilinear',
    #         padding_mode='border',
    #         align_corners=True,
    #     ) # (Nb,5,1,3)
    else:
        nodes  = torch.linspace(ltau.min(),ltau.max(),nnodes)
        normal = NormalizeNodes(ltau.squeeze(),nodes)
        ltnodes = torch.zeros(Nb,1,nnodes,2).to(device=x.device,dtype=x.dtype)
        ltnodes[...,0] = normal.to(x.device,dtype=x.dtype)[None,None,None,:]
        xnodes = F.grid_sample(
            tvbgf,
            ltnodes,
            mode='bicubic',
            padding_mode='border',
            align_corners=True,
        ) # (Nb,5,1,nnodes)
    xnodes = torch.cat([xnodes.reshape(Nb,-1),m.unsqueeze(1),M.unsqueeze(1)],dim=1) # (Nb,5*nnodes+2)
    return xnodes

def node2grid(ltau,xnodes,nnodes=1):
    Nt = ltau.numel()
    if nnodes>=Nt:
        return xnodes
    tvbgf = xnodes[:,:5*nnodes].reshape(-1,5,nnodes)[:,:,None,:].clone() # (Nb,5,1,nnodes)
    Nb = xnodes.size(0)
    m  = xnodes[:,5*nnodes  ]
    M  = xnodes[:,5*nnodes+1]
    if nnodes==1:
        xgrid = tvbgf.repeat(1,1,1,Nt)
    else :
        mode = 'bicubic' if (nnodes>=4) else 'bilinear'
        ltnodes = torch.linspace(ltau.min(),ltau.max(),nnodes).to(xnodes.device,dtype=xnodes.dtype)
        normal = NormalizeNodes(ltnodes,ltau.squeeze())
        # print(normal)
        ltgrid = torch.zeros(Nb,1,Nt,2).to(device=xnodes.device,dtype=xnodes.dtype)
        # print(ltgrid.shape,normal.shape)
        ltgrid[...,0] = normal[None,None,:]
        # print(tvbgf)
        xgrid = F.grid_sample(
            tvbgf,
            ltgrid,
            mode=mode,
            padding_mode='border',
            align_corners=True,
        ) # (Nb,5,1,Nt)
    # else:
    #     ltnodes = torch.linspace(-ltau.min(),ltau.max(),nnodes).to(x.device,dtype=x.dtype)
    #     normal = NormalizeNodes(ltnodes,ltau)
    #     ltgrid = torch.zeros(Nb,5,Nt,2).to(device=x.device,dtype=x.dtype)
    #     ltgrid[...,0] = normal[None,None,:,None]
    #     xgrid = F.grid_sample(
    #         tvbgf,
    #         ltgrid,
    #         mode='bicubic',
    #         padding_mode='border',
    #         align_corners=True,
    #     ) # (Nb,5,1,Nt)
    # print(xgrid[0,1,0,:])
    # print(xgrid.shape,ltgrid.shape)
    x = torch.cat([xgrid.reshape(Nb,-1),m.unsqueeze(1),M.unsqueeze(1)],dim=1) # (Nb,5*Nt+2)
    return x

def spline_smooth(x, y, s=None, k=3, adaptive=False):
    """
    Spline smooth

    Paramerter:
    ------------
    x, y: data point
    s: smooth factor, None for interpolation, the larger the smoother
    k: spline order
    adaptive: whether to use adaptive smoothing
    
    Return:
    ------------
    y_smooth: smoothed data
    spline: spline object
    """
    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
    # enforce increasing x
    if not np.all(np.diff(x) > 0):
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
    else:
        x_sorted = x
        y_sorted = y
    
    if adaptive and s is None:
        # adaptive smoothing
        # estimating s based on the noise level
        residuals = np.diff(y_sorted, 2)  # estimate noise by second derivative
        noise_level = np.std(residuals) / np.sqrt(6)  # standard deviation of second derivative
        s = len(x_sorted) * (noise_level**2)  # automatically calculate s
    # create spline
    spline = UnivariateSpline(x_sorted, y_sorted, s=s, k=k)
    # compute smoothed values
    y_smooth = spline(x_sorted)
    return y_smooth, spline