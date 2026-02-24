from .needs import *
from .phys import *
from .atoms import atomic_config
from .partition_function import u123
from .math import signclamp, RK4_solver, lagrange_interp,derivative
import pkg_resources
import warnings
from .absorption import continuum_absorption

kB = const.k*1e7
nconsider = 28 # number of components considered in the HSE equation

def ie_pressure(theta, pe, **kwargs):
    '''
    Build the atomospheric pressure distribution following the Ionization Equilibrium (IE) equation
    ---------
    Input:
        theta: torch.Tensor #[Nb,Nt,1] -> theta = 5040./T
        pe   : torch.Tensor #[Nb,Nt,1] -> electron pressure

    Return:
        p: torch.Tensor #[Nb,Nt,1] -> gas pressure
    '''
    # p = torch.zeros(99,*theta.shape)
    device=theta.device
    dtype=theta.dtype
    atomic_properties = kwargs.get('atomic_properties',atomic_config())
    p = dict()
    Nb = theta.shape[0]
    Nt = theta.shape[1]
    T = 5040./theta
    neg_pe = pe<=0
    if torch.any(neg_pe):
        pe = torch.where(neg_pe,1.e-10,pe)
    cmol = molecular_balance(theta) # molecular constant cmol[0]-> H2+, cmol[1]-> H2
    cmol = torch.clamp(cmol, -30, 30)
    g4 = pe*torch.pow(10,cmol[0])
    g5 = pe*torch.pow(10,cmol[1])
    g4 = torch.where(neg_pe,0.,g4)
    g5 = torch.where(neg_pe,0.,g5)

    abundance = torch.zeros(nconsider,1,1,1,device=device,dtype=dtype) # [28,1,1,1] -> abu of 28 compenents
    chi1 = torch.zeros_like(abundance) # first ionization potential of 28 compenents
    chi2 = torch.zeros_like(abundance) # second ionization potential of 28 compenents
    u0 = torch.zeros(nconsider,*theta.shape).to(device=device,dtype=dtype) # [28,Nb,Nt,1] -> partition function of neutral atom for 28 compenents
    u1 = torch.zeros_like(u0) # partition function of ionized atom for 28 compenents
    u2 = torch.zeros_like(u0) # partition function of doubly ionized atom for 28 compenents
    for i in range(nconsider):
        weight, abundance[i], chi1[i], chi2[i] = atomic_properties(i+1)
        u0[i], u1[i], u2[i] = u123(i+1,T)

    g2 = Saha(theta,chi1[1],u0[1],u1[1],pe) # p(H+)/p(H)
    # p[91] = g2
    p['p(H+)/p(H)'] = g2
    g3 = Saha(theta,0.754,1,u0[1],pe) # p(H-)/p(H)
    g3 = torch.clamp(g3,min=1.e-30,max=1.e30)
    g3 = 1/g3
    # p[92] = g3
    p['p(H-)/p(H)'] = g3
    g1 = torch.zeros_like(T)

    for i in range(1,nconsider):
        a = Saha(theta,chi1[i],u0[i],u1[i],pe)
        b = Saha(theta,chi2[i],u1[i],u2[i],pe)
        c = 1+a*(1+b)
        c = signclamp(c,min=1.e-20,max=1.e20)

        # p[i] = abundance[i]/c  # p(Xi)/P(H')
        pi = abundance[i]/c
        p[f"p({atomic_properties.symbol[i]})/p(H')"] = pi
        ss1 = (1+2.*b)
        ss1 = signclamp(ss1,min=1.e-20,max=1.e20)
        ss  = pi*a*ss1

        g1 = g1+ss
    a = 1+g2+g3

    if torch.any(g5<1.e-35):
        raise ValueError("The electron pressure is too small")
    g5 = signclamp(g5,min=1.e-20,max=1.e20)
    b = 2.*(1+g2/g5*g4)
    c = g5
    d = g2-g3
    e = g2/g5*g4
    a = signclamp(a,min=1.e-15,max=1.e15)
    b = signclamp(b,min=1.e-15,max=1.e15)
    c = signclamp(c,min=1.e-15,max=1.e15)
    d = signclamp(d,min=1.e-15,max=1.e15)
    e = signclamp(e,min=1.e-15,max=1.e15)

    c1 = c*b**2+a*d*b-e*a**2
    c2 = 2.*a*e-d*b+a*b*g1
    c3 = -(e+b*g1)
    c1 = signclamp(c1,min=1.e-15,max=1.e15)

    f1 = 0.5*c2/c1
    f1 = -f1+torch.sign(c1)*torch.sqrt(f1**2-c3/c1) # p(H)/p(H')
    # f1 = f1_formal_sol1(g1,g2,g3,g4,g5)
    # f1 = torch.where(f1<0,f1_formal_sol2(g1,g2,g3,g4,g5),f1)
    f5 = (1.-a*f1)/b # P(H2)/P(H')
    f4 = e*f5 # P(H2+)/P(H')
    f3 = g3*f1 # P(H-)/P(H')
    f2 = g2*f1 # P(H+)/P(H')
    fe = torch.abs(signclamp(f2-f3+f4+g1,min=1.e-15,max=1.e15)) # P(e-)/P(H')

    phtot=pe/fe
    
    large_f5 = f5>1.e-4
    pg = torch.zeros_like(f1)
    pg_large_f5 = (pe*(1.+(f1+f2+f3+f4+f5+0.1014)/fe))[large_f5]
    const6=g5/pe*f1**2
    const7=signclamp(f2-f3+g1,min=1.e-15,max=1.e15)
    for i in range(5):
        f5=phtot*const6
        f4 = e*f5
        fe = torch.abs(signclamp(f2-f3+f4+g1,min=1.e-15,max=1.e15))
        phtot = pe/fe
    pg = pe*(1.+(f1+f2+f3+f4+f5+0.1014)/fe)
    pg[large_f5] = pg_large_f5
    # p[0] = f1 # p(H)/p(H')
    p["p(H)/p(H')"] = f1
    pg = signclamp(pg,min=1.e-20,max=1.e20)
    # p[83]=pg # gas pressure
    # p[84]=phtot # p(H')
    # p[85]=f2 # p(H+)/p(H')
    # p[86]=f3 # p(H-)/p(H')
    # p[87]=f4 # p(H2+)/p(H')
    # p[88]=f5 # p(H2)/p(H')
    # p[89]=fe # p(e-)/p(H')
    # p[90]=pe/(kB*T) # n(e)=pe/kT
    p["p(H')"] = phtot
    p["p(e-)/p(H')"] = fe
    p["n(e)"] = pe/(kB*T)
    p["p(H2)/p(H')"] = f5
    p["p(H2+)/p(H')"] = f4
    p["p(H-)/p(H')"] = f3
    p["p(H+)/p(H')"] = f2
    p["pg"] = pg

    debug = kwargs.get('debug', False)
    if debug:
        debug_log = dict(
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            f5=f5,
            fe=fe,
            phtot=phtot,
            pg=pg,
            a=a,b=b,c=c,d=d,e=e,
            g1=g1,g2=g2,g3=g3,g4=g4,g5=g5,
        )
        return p,debug_log
    return p

def modify_pe(theta,pe,pg,**kwargs):
    device=theta.device
    dtype=theta.dtype
    atomic_properties = kwargs.get('atomic_properties',atomic_config())
    T = 5040./theta
    neg_pe = pe<=0
    if torch.any(neg_pe):
        pe = torch.where(neg_pe,1.e-15,pe)
    cmol = molecular_balance(theta) # molecular constant cmol[0]-> H2+, cmol[1]-> H2
    cmol = torch.clamp(cmol, -30, 30)
    g4 = pe*torch.pow(10,cmol[0])
    g5 = pe*torch.pow(10,cmol[1])
    g4 = torch.where(neg_pe,0.,g4)
    g5 = torch.where(neg_pe,0.,g5)

    abundance = torch.zeros(nconsider,1,1,1,device=device,dtype=dtype) # [28,1,1,1] -> abu of 28 compenents
    chi1 = torch.zeros_like(abundance) # first ionization potential of 28 compenents
    chi2 = torch.zeros_like(abundance) # second ionization potential of 28 compenents
    u0 = torch.zeros(nconsider,*theta.shape).to(device=device,dtype=dtype) # [28,Nb,Nt,1] -> partition function of neutral atom for 28 compenents
    u1 = torch.zeros_like(u0) # partition function of ionized atom for 28 compenents
    u2 = torch.zeros_like(u0) # partition function of doubly ionized atom for 28 compenents
    for i in range(nconsider):
        weight, abundance[i], chi1[i], chi2[i] = atomic_properties(i+1)
        u0[i], u1[i], u2[i] = u123(i+1,T)

    g2 = Saha(theta,chi1[1],u0[1],u1[1],pe) # p(H+)/p(H)
    g3 = Saha(theta,0.754,1,u0[1],pe) # p(H-)/p(H)
    g3 = torch.clamp(g3,min=1.e-30,max=1.e30)
    g3 = 1/g3
    g1 = torch.zeros_like(T)

    for i in range(1,nconsider):
        a = Saha(theta,chi1[i],u0[i],u1[i],pe)
        b = Saha(theta,chi2[i],u1[i],u2[i],pe)
        c = 1+a*(1+b)
        c = signclamp(c,min=1.e-20,max=1.e20)

        pi = abundance[i]/c
        ss1 = (1+2.*b)
        ss1 = signclamp(ss1,min=1.e-20,max=1.e20)
        ss  = pi*a*ss1
        g1 = g1+ss
    a = 1+g2+g3
    b = 2.*(1+g2/g5*g4)
    c = g5
    d = g2-g3
    e = g2/g5*g4
    a = signclamp(a,min=1.e-15,max=1.e15)
    d = signclamp(d,min=1.e-15,max=1.e15)

    c1 = c*b**2+a*d*b-e*a**2
    c2 = 2.*a*e-d*b+a*b*g1
    c3 = -(e+b*g1)

    f1 = 0.5*c2/c1
    f1 = -f1+torch.sign(c1)*torch.sqrt(f1**2-c3/c1) # p(H)/p(H')
    f5 = (1.-a*f1)/b # P(H2)/P(H')
    f4 = e*f5 # P(H2+)/P(H')
    f3 = g3*f1 # P(H-)/P(H')
    f2 = g2*f1 # P(H+)/P(H')
    fe = torch.abs(signclamp(f2-f3+f4+g1,min=1.e-30,max=1.e30)) # P(e-)/P(H')

    phtot=pe/fe
    
    large_f5 = f5>1.e-4
    pe_new = torch.zeros_like(f1)
    pe_large_f5 = (pg/(1.+(f1+f2+f3+f4+f5+0.1014)/fe))[large_f5]
    const6=g5/pe*f1**2
    const7=signclamp(f2-f3+g1,min=1.e-15,max=1.e15)
    for i in range(5):
        f5=phtot*const6
        f4 = e*f5
        fe = const7+f4
        phtot = pe/fe
    pe_new = pg/(1.+(f1+f2+f3+f4+f5+0.1014)/fe)
    pe_new[large_f5] = pe_large_f5
    pe_new = torch.where(pe<=0,1e-15,pe_new)
    return pe_new

def relax_pe(T,Pg,maxiters=20):
    prec = 1.e-5
    theta = 5040/T
    size = T.shape
    net = torch.load(
        pkg_resources.resource_filename(__name__, 'data/pefrompg.pkl'),
        weights_only=False
    ).to(device=T.device,dtype=T.dtype)
    pe0 = torch.pow(10,net(T.ravel()/1e3,Pg.ravel())).reshape(size)
    for i in range(maxiters):
        pe1 = modify_pe(theta,pe0,Pg)
        diff = torch.abs((pe1-pe0)/pe0)
        if torch.all(diff<=prec):
            break
        else:
            pe0 = (pe0+pe1)/2
    if torch.any(diff>prec):
        warnings.warn(f'Pressure relaxation did not converge after {maxiters} iterations.')
    return pe1

def hse_pressure(ltau,kappac,B,Pg_top,hc_mu=1):
    """
    Parameters:
        ltau    : log(tau) # ( 1,Nt,1), expand from small to large
        kappac  : kappa    # (Nb,Nt,1)
        B       : Bmag     # (Nb,Nt,1)
        Pg_top  : top boundary pressure # (Nb,1)
        hc_mu   : cos(hc)  # default value is 1 cos(heliocentric angle)
        AP      : atomic properties
    Returns:
        Pg    # (Nb,Nt,1)
    """
    Nt = ltau.squeeze().size(0)
    g = 2.74e4*hc_mu
    Pmag = B**2/(8*np.pi)
    dPmag = derivative(Pmag,ltau,dim=1)
    lkapp = torch.log10(kappac)
    def rfunc(ltaui,Pi):
        i = torch.searchsorted(ltau.squeeze(),ltaui).clamp(min=1,max=Nt-2)
        h = i-1
        j = i+1
        lti = torch.tensor([ltaui]).unsqueeze(0)
        lkappi = lagrange_interp(
            # torch.tensor([ltaui]).unsqueeze(0),            
            ltau[:,i],
            lkapp[:,h],lkapp[:,i],lkapp[:,j],
            ltau[:,h],ltau[:,i],ltau[:,j]
        )
        dPmagi = lagrange_interp(
            ltau[:,i],
            # torch.tensor([ltaui]).unsqueeze(0),
            dPmag[:,h],dPmag[:,i],dPmag[:,j],
            ltau[:,h],ltau[:,i],ltau[:,j]
        )
        kappai = torch.pow(10,lkappi)
        ret = np.log(10)*torch.pow(10,ltaui)*g/kappai-dPmagi
        return ret
    Pg_hse = torch.stack(RK4_solver(rfunc)(ltau.squeeze(), Pg_top),dim=1)
    return Pg_hse

def hse_pressure_new(ltau,T,Pg_top,hc_mu=1,AP=atomic_config()):
    """
    Calculate the HSE pressure
    Parameters:
        ltau    : log(tau) # ( 1,Nt,1), expand from small to large
        T       : temperature # (Nb,Nt,1)
        Pg_top  : top boundary pressure # (Nb,1)
    Returns:
        Pg    # (Nb,Nt,1)
    """
    # Rg = const.R*1e7
    NA = const.N_A
    Nt = ltau.squeeze().size(0)
    pg = torch.zeros_like(T)
    kappa = torch.zeros_like(T)
    wref = torch.tensor([5000.],dtype=T.dtype,device=T.device)[None,None,:]
    # pe = torch.zeros_like(T)
    pg[:,0] = Pg_top
    prec = 1.e-5
    maxiters = 20
    lt = ltau.squeeze()
    tau = torch.pow(10,lt)
    amw = AP.amw
    abusum = AP.abusum
    g = hc_mu*2.7414e4
    x = torch.zeros_like(lt)
    x[:-1] = g*(lt[1:]-lt[:-1])*np.log(10)
    Tei = T[:,0:1]
    pgi = Pg_top.unsqueeze(1)
    pei = relax_pe(Tei,pgi)
    ppi = ie_pressure(5040/Tei,pei)
    kac = continuum_absorption(wref,Tei,pei,ppi)
    mu = amw/(abusum+ppi["p(e-)/p(H')"])
    kappa[:,0:1] = kac*NA/mu
    for i in range(1,Nt):
        pgj = (pg[:,i-1]+x[i-1]*tau[i-1]/kappa[:,i-1]).unsqueeze(1)
        Tei = T[:,i:i+1]
        pei = relax_pe(Tei,pgj)
        ppi = ie_pressure(5040/Tei,pei)
        kac = continuum_absorption(wref,Tei,pei,ppi)
        mu  = amw/(abusum+ppi["p(e-)/p(H')"])
        kappa[:,i:i+1] = kac*NA/mu
        pg[:,i]=pg[:,i-1]+x[i-1]*(tau[i-1]/kappa[:,i-1]+tau[i]/kappa[:,i])/2
        dif = 2*torch.abs((pg[:,i]-pgj[:,0]))/(pg[:,i]+pgj[:,0])
        for nit in range(maxiters):
            if torch.all(dif<=prec):
                break
            else:
                pgj = pg[:,i:i+1]
                pei = relax_pe(Tei,pgj)
                ppi = ie_pressure(5040/Tei,pei)
                kac = continuum_absorption(wref,Tei,pei,ppi)
                mu  = amw/(abusum+ppi["p(e-)/p(H')"])
                kappa[:,i:i+1] = kac*NA/mu
                pg[:,i]=pg[:,i-1]+x[i-1]*(tau[i-1]/kappa[:,i-1]+tau[i]/kappa[:,i])/2
                dif = 2*torch.abs((pg[:,i]-pgj[:,0]))/(pg[:,i]+pgj[:,0])
        if torch.any(dif>prec):
            warnings.warn(f'HSE relaxation did not converge after {maxiters} iterations.')
    return pg
