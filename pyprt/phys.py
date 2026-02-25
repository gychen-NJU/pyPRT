from .needs import *

def Planck_Blambda(T, lambdas):
    lambdas = lambdas*1e-8
    h = const.h*1e7
    c = const.c*1e2
    k = const.k*1e7
    exponent = h*c/(lambdas*k*T)
    numerator = 2*h*c**2/lambdas**5
    denominator = torch.expm1(exponent)
    small_value_mask = exponent<1e-3
    RJ_approx = 2*c*k*T/lambdas**4
    ret = torch.where(small_value_mask, RJ_approx, numerator/denominator)
    return ret

def Planck_Bnu(T, nu):
    h = const.h*1e7
    c = const.c*1e2
    k = const.k*1e7
    numerator = 2*h*nu**3/c**2
    exponent  = h*nu/(k*T)
    denominator = torch.expm1(exponent)
    small_value_mask = exponent<1e-6
    B_nu_approx = (2*nu**2*k*T)/c**2
    ret = torch.where(small_value_mask, B_nu_approx, numerator/denominator)
    return ret

def Unnormalized_strengths(j1,j2):
    if np.abs(j1-j2)>1:
        raise ValueError(f"Ju-Jl={j2-j1} is inconsistent with the selection rule")
    m1list = np.arange(-j1,j1+1,1)
    m2list = np.arange(-j2,j2+1,1)
    # pi_p component
    Sp = []
    m2p = []
    for m2 in range(max(m1list.min(), m2list.min()), min(m1list.max(),m2list.max())+1):
        if np.isclose(j2-j1,1):
            Sp.append(2*(j2**2-m2**2))
        elif np.isclose(j2-j1,0):
            Sp.append(2*m2**2)
        elif np.isclose(j2-j1,-1):
            Sp.append(2*(j1**2-m2**2))
        m2p.append(m2)
    m2p = np.array(m2p)
    Sp  = np.array(Sp)
    Sb = []
    m2b = []
    for m2 in range(max((m1list+1).min(), m2list.min()), min((m1list+1).max(),m2list.max())+1):
        if np.isclose(j2-j1,1):
            Sb.append((j2+m2)*(j1+m2))
        elif np.isclose(j2-j1,0):
            Sb.append((j2+m2)*(j2-m2+1))
        elif np.isclose(j2-j1,-1):
            Sb.append((j1-m2)*(j2-m2+2))
        m2b.append(m2)
    m2b = np.array(m2b)
    Sb  = np.array(Sb)
    Sr = []
    m2r = []
    for m2 in range(max((m1list-1).min(), m2list.min()), min((m1list-1).max(),m2list.max())+1):
        if np.isclose(j2-j1,1):
            Sr.append((j2-m2)*(j1-m2))
        elif np.isclose(j2-j1,0):
            Sr.append((j2-m2)*(j2+m2+1))
        elif np.isclose(j2-j1,-1):
            Sr.append((j1+m2)*(j2+m2+2))
        m2r.append(m2)
    m2r = np.array(m2r)
    Sr  = np.array(Sr)
    return Sb,Sp,Sr,m2b,m2p,m2r


def Apply_Macroturbulence(Stokes, vmac, lambdas):
    """
    Parameters:
    ------------
    Stokes: Tensor (Nb,Nw,4)
    vmac: Tensor (Nb,1) [km/s]
    lambdas: Tensor (1,1,Nw)
    """
    device = Stokes.device
    dtype  = Stokes.dtype
    lambdai = lambdas.squeeze(1).unsqueeze(-1) # (1,Nw,1)
    dl = torch.diff(lambdas.squeeze()).min()
    c = const.c
    Stokes_obs = Stokes.clone()
    lambdac = lambdai.mean().item()
    s_mac = lambdac*vmac*1e3/c # (Nb,1)
    width = s_mac.max().item()
    lambdaj = torch.arange(-3*width,3*width+dl,dl)[None,:].to(device=device,dtype=dtype) # (1,M)
    G_mac = 1/np.sqrt(np.pi)/s_mac*torch.exp(-lambdaj**2/s_mac**2)*dl # (Nb, M)
    G_mac = G_mac/G_mac.sum(dim=1,keepdim=True)
    N = Stokes.size(1)
    W = lambdaj.size(1)
    Nb = Stokes.size(0)
    if W<=1:
        return Stokes_obs
    i_indices = torch.arange(N).to(device=device).unsqueeze(0)
    j_indices = torch.arange(N).to(device=device).unsqueeze(1)
    center = W // 2
    diff = i_indices - j_indices + center
    mask = (diff >= 0) & (diff < W)
    Toeplitz = torch.zeros(Nb,N,N).to(device=device,dtype=dtype) # (Nb,Nw,Nw)
    # print(G_mac.shape, vmac.shape)
    Toeplitz[:,mask]=G_mac[:,diff[mask]]
    for i in range(4):
        Stokes_obs[:,:,i:i+1] = torch.matmul(Toeplitz, Stokes_obs[:,:,i:i+1])
    return Stokes_obs

def Apply_Macroturbulence_Conv(Stokes, vmac, lambdas):
    """
    Parameters:
    ------------
    Stokes: Tensor (Nb,Nw,4)
    vmac: Tensor (Nb,1) [km/s]
    lambdas: Tensor (1,1,Nw)
    """
    device = Stokes.device
    dtype  = Stokes.dtype
    if torch.any(torch.isclose(vmac,torch.zeros_like(vmac),atol=1e-3)):
        return Stokes
    Nb = Stokes.size(0)
    Nc = Stokes.size(2)
    dl = torch.diff(lambdas.squeeze()).min()
    c = const.c
    lambdac = lambdas.squeeze().mean().item()
    s_mac = lambdac*vmac*1e3/c # (Nb,1)
    width = s_mac.max().item()
    sigma_pixels = width/dl.item()
    kernel_size  = int(2*np.ceil(3*sigma_pixels)+1)
    kernel_size  = kernel_size if kernel_size%2==1 else kernel_size+1
    radius = kernel_size // 2
    x = torch.arange(-radius,radius+1,device=device,dtype=dtype)[None,None,None,:] # (1,1,1,Nk)
    sigma = (s_mac/dl)[...,None,None] # (Nb,1,1,1)
    kernels =  torch.exp(-x**2 / (2 * sigma**2)) # (Nb,1,1,Nk)
    kernels = kernels/kernels.sum(dim=-1,keepdim=True)
    kernels = kernels.repeat(1,Nc,1,1) # (Nb,Nc,1,Nk)
    kernels = kernels.reshape(Nb*Nc,1,-1)
    spectra = Stokes.clone().permute(0,2,1) # (Nb,Nc,Nw)
    spectra = spectra.reshape(1,Nb*Nc,-1) # (1,Nb*Nc,Nw)
    spectra_padded = F.pad(spectra, (radius, radius), mode='reflect')
    convolved = F.conv1d(spectra_padded, kernels, padding=0,groups=Nb*Nc)
    Stokes_obs = convolved.reshape(Nb,Nc,-1).permute(0,2,1)
    Stokes_obs = torch.where(torch.isclose(vmac,torch.zeros_like(vmac),atol=1e-3)[:,:,None],Stokes,Stokes_obs)
    return Stokes_obs

def Saha(theta, chi, uj, ui, Pe):
    """
    Saha Inoization equation

    Parameters:
    ------------
    theta: 1ev/(ln(10)*kB*T)
    chi  : ionization potential in unit [eV]
    uj   : patition function of i+1 state
    ui   : patition function of i state
    Pe   : electron pressure in unit dyn/cm^2

    Return:
    nj/ni = 2uj/ui*(2 pi me k T)^1.5/h^3 * exp(-chi/kT)/Ne
    """
    # print(uj.device,ui.device,theta.device)
    ret = uj/ui*torch.pow(10, 9.0805126-theta*chi)/(Pe*theta**2.5)
    return ret


def molecular_balance(theta):
    '''
    Calculate the molecular constant cmol[0]-> H2+, cmol[1]-> H2
    ---------
    Input:
        theta: torch.Tensor #[Nb,Nt,1] -> theta = 5040./T

    Return:
        cmol: torch.Tensor #[Nb,Nt,2] -> cmol[0]-> H2+, cmol[1]-> H2
    '''
    cmol = torch.zeros(2,*theta.shape).to(device=theta.device, dtype=theta.dtype)
    cmol[0] = -11.206998+theta*(2.7942767+theta*(7.9196803e-2-theta*2.4790744e-2)) # H2+
    cmol[1] = -12.533505+theta*(4.9251644+theta*(-5.6191273E-2+theta*3.2687661E-3)) # H2
    return cmol
