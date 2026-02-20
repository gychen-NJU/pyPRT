from pprint import pformat
from .needs import *
# from .atoms import atomic_properties
from .partition_function import u123
from .phys import Saha

h = const.h*1e7
kB = const.k*1e7
c = const.c*1e2

ghel = [1.,3.,1.,9.,9.,3.,3.,3.,1.,9.,20.,3.]
chihel = [0.,19.819,20.615,20.964,20.964,21.217,21.217,22.718,22.920,23.007,23.073,23.087]
Tgrid_C = [
    3.9999e3,5e3,6e3,7e3,8e3,9e3,1e4,1.1e4,1.2e4,1.3e4,
    1.4e4,1.5e4,1.6e4,1.7e4,1.8e4,1.9e4,2e4,2.1e4,2.2e4,
    2.3e4,2.4e4,2.5e4,2.6e4,2.7e4,2.8e4,2.9e4,3.00001e4
    ]
fgrid_C = [
    1.728,6.660,9.964,12.340,14.134,15.540,16.676,17.614,18.402,
    19.076,19.662,20.172,20.626,21.030,21.394,21.722,22.022,22.296,
    22.548,22.78,22.996,23.196,23.384,23.56,23.724,23.878,24.024
    ]
fgrid_Na = [
    1.638,3.914,5.461,6.586,7.448,8.130,8.686,9.152,9.546,9.887,10.184,
    10.448,10.682,10.894,11.084,11.258,11.418,11.566,11.704,11.830,11.950,
    12.062,12.166,12.264,12.358,12.448,12.532
]
fgrid_Mg = [
    1.784,5.268,7.618,9.318,10.61,11.628,12.452,13.134,13.71,14.204,14.632,15.008,15.342,
    15.64,15.906,16.15,16.37,16.574,16.76,16.934,17.094,17.244,17.384,17.514,17.638,17.754,17.866
]
wgrid_C = [
    0.,1.1005e-5,1.1005e-5,1.2395e-5,1.2395e-5,1.4445e-5,1.4445e-5,2.178e-5,
    2.912e-5,3.4625e-5,3.4625e-5,4.7295e-5,4.7295e-5,5.4785e-5,7.1e-5
    ]
wgrid_Na = [
    0.,1.2e-5,1.5e-5,1.6e-5,1.7e-5,1.8e-5,1.95e-5,2.1e-5,2.2e-5,2.3e-5,2.4125e-5,2.4125e-5,
    2.75e-5,4.0845e-5,4.0845e-5,6.366e-5,8.1455e-5,8.1455e-5,8.9455e-5,8.9455e-5,9.85e-5
]
wgrid_Mg = [
    1.3999e-5,1.5e-5,1.6215e-5,1.6215e-5,2.5135e-5,2.5135e-5,2.75e-5,3.7565e-5,3.7565e-5,4.8845e-5,
    4.8845e-5,6.5495e-5,6.5495e-5,7.2345e-5,7.2345e-5,7.2915e-5,7.2915e-5,8.1135e-5,8.1135e-5,9.e-5
]
cmesh_C = np.array([
    [-2.75, -3.02, -1.16, -1.17, 1.33, 1.32, 8.05, 7.79, 7.66, 7.68, 8.12, 8.07, 8.28, 8.44, 8.14],
    [-2.75, -3.02, -1.48, -1.49, 0.66, 0.65, 5.99, 5.73, 5.61, 5.63, 5.86, 5.79, 5.95, 5.97, 5.67],
    [-2.75, -3.02, -1.68, -1.69, 0.22, 0.20, 4.58, 4.32, 4.21, 4.20, 4.36, 4.25, 4.36, 4.35, 4.02],
    [-2.75, -3.02, -1.83, -1.84, -0.10, -0.11, 3.56, 3.30, 3.20, 3.18, 3.27, 3.10, 3.19, 3.08, 2.83],
    [-2.75, -3.02, -1.94, -1.95, -0.33, -0.34, 2.78, 2.52, 2.38, 2.36, 2.43, 2.22, 2.34, 2.25, 1.93],
    [-2.75, -3.02, -2.08, -2.09, -0.65, -0.67, 1.67, 1.41, 1.27, 1.20, 1.24, 1.03, 1.08, 0.97, 0.66],
    [-2.75, -3.02, -2.27, -2.29, -1.10, -1.12, 0.11, -0.14, -0.21, -0.42, -0.40, -0.64, -0.64, -0.78, -1.09],
    [-2.75, -3.02, -2.37, -2.38, -1.38, -1.41, -0.75, -0.98, -1.16, -1.29, -1.28, -1.57, -1.55, -1.70, -2.01],
    [-2.75, -3.02, -2.51, -2.53, -1.88, -1.91, -1.70, -1.91, -2.10, -2.24, -2.24, -2.55, -2.54, -2.70, -3.01]
]).T # -(log(kappa)+20)
cmesh_Na = np.array([
    [8.70, 8.70, 8.70, 8.70, 8.65, 8.60, 8.50, 8.40],  
    [8.78, 8.78, 8.78, 8.75, 8.72, 8.61, 8.48, 8.26],  
    [8.99, 8.99, 8.99, 8.94, 8.86, 8.62, 8.42, 8.10],  
    [9.13, 9.13, 9.12, 9.01, 8.91, 8.63, 8.38, 8.04],  
    [9.41, 9.37, 9.31, 9.15, 9.00, 8.62, 8.34, 7.98],  
    [10.11, 9.93, 9.72, 9.33, 9.06, 8.59, 8.29, 7.89], 
    [10.62, 10.10, 9.72, 9.25, 8.95, 8.47, 8.17, 7.78],
    [9.95, 9.72, 9.47, 9.06, 8.78, 8.33, 8.06, 7.67],  
    [9.31, 9.23, 9.11, 8.85, 8.63, 8.23, 7.96, 7.59],  
    [9.00, 8.96, 8.88, 8.68, 8.49, 8.13, 7.87, 7.52],  
    [8.87, 8.82, 8.75, 8.57, 8.39, 8.02, 7.79, 7.44],  
    [10.22, 9.67, 9.31, 8.84, 8.55, 8.10, 7.83, 7.45], 
    [9.96, 9.43, 9.07, 8.60, 8.31, 7.87, 7.61, 7.24],  
    [9.26, 8.73, 8.34, 7.93, 7.63, 7.19, 6.98, 6.62],  
    [11.45, 10.52, 9.86, 9.03, 8.52, 7.79, 7.35, 6.88],
    [10.76, 9.83, 9.18, 8.37, 7.86, 7.12, 6.70, 6.23], 
    [10.43, 9.48, 8.84, 8.02, 7.52, 6.87, 6.36, 5.88], 
    [10.95, 9.92, 9.24, 8.33, 7.77, 6.94, 6.49, 5.96], 
    [10.82, 9.80, 9.12, 8.20, 7.64, 6.82, 6.36, 5.83], 
    [11.32, 10.18, 9.41, 8.41, 7.79, 6.93, 6.43, 5.87],
    [11.18, 10.04, 9.27, 8.28, 7.66, 6.78, 6.29, 5.73] 
]) # -log(kappa)-10
cmesh_Mg = np.array([
    [-0.57, -0.87, -1.20, -1.72, -2.05, -2.53, -2.82, -3.18],  
    [-1.55, -1.61, -1.71, -1.99, -2.24, -2.67, -2.92, -3.26],  
    [-2.08, -2.10, -2.14, -2.30, -2.48, -2.83, -3.06, -3.36],  
    [-0.20, -0.89, -1.34, -1.92, -2.26, -2.74, -3.01, -3.33],  
    [-0.75, -1.44, -1.89, -2.47, -2.82, -3.30, -3.55, -3.85],  
    [2.14, 1.01, 0.25, -0.74, -1.36, -2.24, -2.73, -3.26],     
    [2.01, 0.88, 0.14, -0.85, -1.47, -2.35, -2.82, -3.36],     
    [2.02, 0.88, 0.07, -0.98, -1.65, -2.59, -3.11, -3.67],     
    [3.03, 1.56, 0.57, -0.69, -1.45, -2.48, -3.03, -3.63],     
    [2.74, 1.26, 0.27, -0.97, -1.68, -2.73, -3.34, -3.94],     
    [2.79, 1.30, 0.32, -0.95, -1.68, -2.73, -3.33, -3.93],     
    [2.49, 1.00, 0.00, -1.22, -2.04, -3.10, -3.66, -4.28],     
    [2.71, 1.19, 0.17, -1.12, -1.90, -3.00, -3.58, -4.22],     
    [2.59, 1.08, 0.07, -1.23, -2.02, -3.12, -3.69, -4.34],     
    [2.75, 1.23, 0.20, -1.10, -1.91, -3.02, -3.63, -4.29],     
    [2.74, 1.22, 0.18, -1.12, -1.92, -3.04, -3.64, -4.30],     
    [3.75, 2.05, 0.88, -0.59, -1.49, -2.77, -3.43, -4.17],     
    [3.67, 1.94, 0.78, -0.70, -1.60, -2.88, -3.56, -4.30],     
    [3.78, 1.98, 0.82, -0.68, -1.60, -2.88, -3.56, -4.30],     
    [3.61, 1.87, 0.69, -0.81, -1.73, -3.00, -3.69, -4.43]      
]) # -log(kappa)-20

def continuum_absorption(lambdas,T,Pe,p,**kwargs):
    """
    Calculate the atmospheric absorption coefficient. kappaC/N(H')

    Args:
        lambdas: torch.Tensor #[1,1,Nw], wavelangth in unit [Angstrum]
        T: torch.Tensor #[Nb,Nt,1], temperature in unit [K]
        Pe: torch.Tensor #[Nb,Nt,1], electron pressure in unit [dyn]
        p: torch.Tensor #[99,Nb,Nt,1],  pressure in unit [dyn]
    return:
        kappa: torch.Tensor #[Nb,Nt,Nw], absorption coefficient in unit [cm^-1]
    """
    refidx = kwargs.get('refidx',1) # refractive index
    debug  = kwargs.get('debug', False)
    Nw = lambdas.size(2)
    Nb = T.size(0)
    Nt = T.size(1)
    wavs_cm = lambdas*1e-8 # centimeter
    wavs_mu = lambdas*1e-4 # micron
    wavs_si = wavs_cm*1e2  # meters
    wavs_km = wavs_cm*1e5  # km
    wavs3   = wavs_cm**3
    freq    = c/(wavs_cm*refidx)
    lnfreq  = torch.log(freq)
    theta = 5040/T
    T25 = torch.pow(T,2.5)
    x0 = wavs_mu # x10000
    x1 = wavs3 # x10001
    x2 = lambdas # 10002
    x3 = h*c/(kB*wavs_cm) # hc/(kT*lambda)  x10003
    x4 = wavs_si # x10004
    x5 = wavs_km # x10005
    x6 = 1e-34*x2**2 # x10006
    deltak = 911.3/(x2*refidx)
    ephot = deltak
    ey = torch.pow(ephot,(0.43+0.6*torch.log10(ephot+10.)))
    divi1 = 1.+(5.9856e-2-(3.4916e-4*ey)/(1.+1.e-2*ey))*ephot**.83333333
    m0 = torch.sqrt(1/ephot).to(dtype=torch.int)+1
    l0 = torch.where(m0<4,4,m0)
    t1 = 1.e-8/wavs_cm
    t2 = t1**2
    t3 = t2**2
    scat1 = t3*(5.799e-13+1.422e-6*t2+2.784*t3)
    scat2 = t3*(8.14e-13+1.28e-6*t2+1.61*t3)
    scat3 = 5.484e-14*t3*(1.+2.44e5*t2+5.94e-10*t2/(x2**2-2.9e5))**2

    kappa = torch.zeros(Nb,Nt,Nw).to(device=T.device,dtype=T.dtype)
    # ======================================================
    # H-
    # ======================================================
    f1 = x3/T
    z1 = torch.where(f1<1.e-3,f1,1-torch.exp(-f1))
    flag_5 = (wavs_cm>1.64189e-4)
    flag_69 = ((wavs_cm<1.64189e-4) & (wavs_cm>1.42e-4))
    x = x5
    cbfree = 1.e-17*(6.80133e-3+x*(1.78708e-1+x*(1.6479e-1-x*(2.04842e-2-5.95244e-4*x))))
    x = 16.419-x5
    cbfree = torch.where(
        flag_69,
        1.e-17*x*(2.69818e-1+x*(2.2019e-1-x*(4.11288e-2-2.73236e-3*x))),
        cbfree
        )
    hminus = cbfree*p["p(H-)/p(H)"]*z1/Pe
    b1 = 11.924-5.939*theta
    c1 = 7.0355-theta*3.4592e-1
    c2 = x5*(-4.0192e-1+theta*c1)
    b2 = x4*(-3.2062+theta*b1+c2)
    b3 = 2.7039e-2*theta-1.1493e-2+b2
    b4 = 5.3666e-3+theta*b3
    hminus = torch.where(flag_5.expand_as(kappa),(hminus+1.e-26*b4)*Pe*p["p(H)/p(H')"],hminus)
    if debug:
        kappa_dict = {"kappa(H-)":hminus}
    del flag_5,flag_69
    # ======================================================
    # He-
    # ======================================================
    flag_73 = (deltak>0.3) | (T<1.5e3) | (T>1.68e4)
    flag_71 = T<9.2e3
    if not torch.all(flag_71):
        a1 = 2.46e-4+wavs_cm*(-1.26e+1+5.67e+6*wavs_cm)
        b1 = theta*(-5.92e-4+wavs_cm*(5.12e+1+1.3e+7*wavs_cm))
        c1 = (theta*theta)*(1.45e-2-wavs_cm*(8.3e+1+1.8e+6*wavs_cm))
        helmin = (a1+b1+c1)*1.e-26
    else:
        helmin = torch.zeros_like(hminus)

    a1 = 9.5114e-9-T*2.3544e-13
    a2 = -1.0754e-4+T*a1
    a3 = .49245+T*a2
    helmin = torch.where(flag_71,x6*a3,helmin)

    helmin = helmin*Pe*p["p(He)/p(H')"]
    helmin = torch.where(flag_73,0,helmin)
    if debug:
        kappa_dict["kappa(He-)"] = helmin
    del flag_73,flag_71

    # ======================================================
    # Cl-
    # ======================================================
    clmin = torch.zeros_like(helmin)
    if debug:
        kappa_dict["kappa(Cl-)"] = clmin

    # ======================================================
    # H I
    # ======================================================  
    flag_10 = m0>12
    X2 = um(12,T)
    X3 = um(1,T)
    sum = torch.zeros_like(X2).expand(Nb,Nt,Nw).clone()
    g = list()
    z3 = list()
    for i in range(1,13):
        uuu = um(i,T)
        z3_i = torch.exp(uuu-X3)/i**3
        divi2 = 1.+(1.72826e-1-3.45652e-1/(ephot*i**2))*ephot**.333333333
        g_i = divi2/divi1
        # print(sum.shape,(g_i*z3_i).shape)
        sum += torch.where(m0<=i,g_i*z3_i,0)
        g.append(g_i)
        z3.append(z3_i)
    g  = torch.stack(g,dim=0)
    z3 = torch.stack(z3,dim=0)
    z2 = torch.full_like(T,2.)
    z2 = torch.where(T>1.3e4,1.51+3.8e-5*T,z2)
    z2 = torch.where(T>1.62e4,11.41+T*(-1.1428e-3+T*3.52e-8),z2)
    a1 = z2
    z2 = 2.08966e-2*x1*z1/a1
    gg = gff(T,x2)
    z4=0.5/X3*(torch.exp(X2-X3)+torch.exp(-X3)*(gg-1.))
    hneutr = z2*(sum+z4)*p["p(H)/p(H')"]
    hneutr = torch.where(flag_10,0,hneutr)
    if debug:
        kappa_dict["kappa(H)"] = hneutr
    del flag_10,g_i,z3_i

    # ======================================================
    # H2-
    # ====================================================== 
    flag_20 = (deltak>0.3) | (T<1.5e3) | (T>1.68e4)
    if torch.all(flag_20):
        h2min = torch.zeros_like(hneutr)
    else:
        h2min = torch.where(
            flag_20,
            0,
            x6*(.88967+T*(-1.4274e-4+T*(1.0879e-8-T*2.5658e-13)))*Pe*p["p(H2)/p(H')"]
        )
    if debug:
        kappa_dict["kappa(H2-)"] = h2min
    del flag_20

    # ======================================================
    # H2+
    # ====================================================== 
    flag_70 = (wavs_cm<3.8e-5) | (wavs_cm>3e-4)
    if torch.all(flag_70):
        h2plus = torch.zeros_like(h2min)
    else:
        exp_part1 = 2.30258509 * theta * (
            7.342e-3 - (-2.409e-15 + (
                1.028e-30 + (-4.23e-46 + (
                    1.224e-61 - 1.351e-77 * freq
                ) * freq) * freq
            ) * freq) * freq
        )
        exp_part2 = -3.0233e3 + (
            3.7797e2 + (-1.82496e1 + (
                3.9207e-1 - 3.1672e-3 * lnfreq
            ) * lnfreq) * lnfreq
        ) * lnfreq
        exponent = exp_part1 + exp_part2
        h2plus = torch.where(
            flag_70,
            0,
            torch.exp(exponent)*1e16*z1*(p["p(H)/p(H')"]*Pe)*(p["p(H+)/p(H')"]/p["p(e-)/p(H')"])/(1.38054*T)
        )
    if debug:
        kappa_dict["kappa(H+)"] = h2plus
    del flag_70

    # ======================================================
    # He
    # ====================================================== 
    heneut = torch.zeros_like(h2plus).clone()
    flag_30 = wavs_cm>8.2610e-5
    flag_36 = wavs_cm>3.6788e-5
    i0 = torch.ones_like(wavs_cm) # 1 1s
    i0 = torch.where(wavs_cm>5.0420e-6,2,i0) # 2 3s
    i0 = torch.where(wavs_cm>2.6003e-5,3,i0) # 2 1s
    i0 = torch.where(wavs_cm>3.1210e-5,4,i0) # 2 3p odd, 2 3p odd
    i0 = torch.where(wavs_cm>3.4210e-5,5,i0) # 2 1p odd, 2 1p odd
    i0 = torch.where(flag_36,8,i0) # 3 3s
    i0 = torch.where(wavs_cm>6.6322e-5,9,i0) # 3 1s
    i0 = torch.where(wavs_cm>7.4351e-5,10,i0) # 3 3p
    i0 = torch.where(wavs_cm>7.8438e-5,11,i0) # 3 1d + 3 3d
    i0 = torch.where(wavs_cm>8.1910e-5,12,i0) # 3 1p odd
    for i in range(1,13):
        if torch.any(i0==i):
            pepa = ghel[i-1]*torch.pow(10,fkny(i,lnfreq)-theta*chihel[i-1])
            heneut += torch.where(i0==i,pepa,0)
        else:
            continue
    flag_31 = m0>12
    sum = torch.zeros_like(g[0].expand(Nb,Nt,Nw))
    for i in range(1,13):
        # if i==1:
        #     print(l0.shape,g[i-1].shape,z3[i-1].shape)
        sum += torch.where(l0.expand(Nb,Nt,Nw)<=i,g[i-1]*z3[i-1],0)
    flag_31 = flag_31 | (theta>3) | (torch.abs(z2)<1.e-25) | (torch.abs(sum+z4)<1.e-25)
    pepo = 4.*torch.pow(10,-10.992*theta)*z2*(sum+z4)
    heneut_temp = torch.where(flag_30,pepo,heneut+pepo)
    heneut = torch.where(flag_31,heneut,heneut_temp)
    heneut = heneut*p["p(He)/p(H')"]
    if debug:
        kappa_dict["kappa(He)"] = heneut
    del flag_30,pepa,pepo,sum,heneut_temp

    # ======================================================
    # Scatter
    # ====================================================== 
    flag_35 = wavs_cm<1.2e-5
    scatt1 = torch.where(flag_35,0,scat1*p["p(H)/p(H')"])
    scatt2 = torch.where(flag_35,0,scat2*p["p(H2)/p(H')"])
    scatt3 = torch.where(flag_35,0,scat3*p["p(He)/p(H')"])
    escatt = 6.653e-25*p["p(e-)/p(H')"]
    if debug:
        kappa_dict["kappa(scatt)"] = scatt1+scatt2+scatt3+escatt
    del flag_35

    # ======================================================
    # C
    # ======================================================
    flag_142 = (T>Tgrid_C[0]) & (T<Tgrid_C[-1])
    flag_50 = wavs_cm>=wgrid_C[-1]
    if torch.all(~flag_142) or torch.all(flag_50):
        carbon = torch.zeros_like(heneut)
    else:
        T_idx = torch.ones_like(T,dtype=torch.int32)
        for i in range(1,len(Tgrid_C)):
            T_idx = torch.where(T>Tgrid_C[i],i+1,T_idx)
        T_idx = torch.where(T_idx>=26,25,T_idx)

        fC_tensor = torch.tensor(fgrid_C, dtype=T.dtype, device=T.device)
        TC_tensor = torch.tensor(Tgrid_C, dtype=T.dtype, device=T.device)
        fh = fC_tensor[T_idx-1]
        fi = fC_tensor[T_idx]
        fj = fC_tensor[T_idx+1]
        Th = TC_tensor[T_idx-1]
        Ti = TC_tensor[T_idx]
        Tj = TC_tensor[T_idx+1]
        y0 = lagrange_interp(T,fh,fi,fj,Th,Ti,Tj)
        l  = torch.clamp(T_idx-1,max=3)
        l  = torch.where(T_idx>4,4,l)
        l  = torch.where(T_idx>6,5,l)
        l  = torch.where(T_idx>11,6,l)
        l  = torch.where(T_idx>16,7,l)
        # print(T_idx.squeeze())
        y  = torch.where(T_idx<=4,(y0-fC_tensor[T_idx-1])/(fC_tensor[T_idx]-fC_tensor[T_idx-1]),y0)
        y  = torch.where(T_idx>4 ,(y0-fgrid_C[4])/2.542,y)
        y  = torch.where(T_idx>6 ,(y0-fgrid_C[6])/3.496,y)
        y  = torch.where(T_idx>11,(y0-fgrid_C[11])/1.850,y)
        y  = torch.where(T_idx>16,(y0-fgrid_C[16])/2.002,y)

        w_idx = torch.ones_like(wavs_cm,dtype=torch.int32)
        for j in range(1,len(wgrid_C)):
            w_idx = torch.where(wavs_cm>wgrid_C[j],j+1,w_idx)
        # print(w_idx.squeeze())
        wC_tensor = torch.tensor(wgrid_C, dtype=T.dtype, device=T.device)
        cC_tensor = torch.from_numpy(cmesh_C).to(device=T.device,dtype=T.dtype)
        y2 = (wavs_cm-wC_tensor[w_idx-1])/(wC_tensor[w_idx]-wC_tensor[w_idx-1])
        y1 = cC_tensor[w_idx-1,l]+y2*(cC_tensor[w_idx,l]-cC_tensor[w_idx-1,l])
        y2 = cC_tensor[w_idx-1,l+1]+y2*(cC_tensor[w_idx,l+1]-cC_tensor[w_idx-1,l+1])
        # print(f"y: {y.squeeze()}")
        # print(f"y1: {y1.squeeze()}")
        # print(f"y2: {y2.squeeze()}")
        # print(f"y*(y1-y2)-y1-20: {(y*(y1-y2)-y1-20.).squeeze()}")
        # print(f"z1: {z1.squeeze()}")
        # print(f"p(C)/p(H'): {p["p(C)/p(H')"].squeeze()}")
        carbon = z1*p["p(C)/p(H')"]*torch.pow(10,y*(y1-y2)-y1-20.)
        carbon = torch.where((~flag_142) | flag_50,0,carbon)
    if debug:
        kappa_dict["kappa(C)"] = carbon
    del flag_50

    # ======================================================
    # Na
    # ======================================================
    flag_61 = wavs_cm<wgrid_Na[-1]
    if torch.all((~flag_61) | (~flag_142)):
        sodium = torch.zeros_like(carbon)
    else:
        fNa_tensor = torch.tensor(fgrid_Na, dtype=T.dtype, device=T.device)
        y0 = lagrange_interp(
            T,
            fNa_tensor[T_idx-1],fNa_tensor[T_idx],fNa_tensor[T_idx+1],
            TC_tensor[T_idx-1],TC_tensor[T_idx],TC_tensor[T_idx+1]
        )
        y = torch.where(T_idx<=2,(y0-fNa_tensor[T_idx-1])/(fNa_tensor[T_idx]-fNa_tensor[T_idx-1]),y0)
        y = torch.where(T_idx>2, (y0-fNa_tensor[2])/1.987,y)
        y = torch.where(T_idx>4, (y0-fNa_tensor[4])/1.238,y)
        y = torch.where(T_idx>6, (y0-fNa_tensor[6])/1.762,y)
        y = torch.where(T_idx>11,(y0-fNa_tensor[11])/0.970,y)
        y = torch.where(T_idx>16,(y0-fNa_tensor[16])/1.114,y)
        l = torch.clamp(T_idx-1,max=1)
        l = torch.where(T_idx>2,2,l)
        l = torch.where(T_idx>4,3,l)
        l = torch.where(T_idx>6,4,l)
        l = torch.where(T_idx>11,5,l)
        l = torch.where(T_idx>16,6,l)
        w_idx = torch.ones_like(wavs_cm,dtype=torch.int32)
        for j in range(1,len(wgrid_Na)):
            w_idx = torch.where(wavs_cm>wgrid_Na[j],j+1,w_idx)
        wNa_tensor = torch.tensor(wgrid_Na, dtype=T.dtype, device=T.device)
        cNa_tensor = torch.from_numpy(cmesh_Na).to(device=T.device,dtype=T.dtype)
        y2 = (wavs_cm-wNa_tensor[w_idx-1])/(wNa_tensor[w_idx]-wNa_tensor[w_idx-1])
        y1 = cNa_tensor[w_idx-1,l]+y2*(cNa_tensor[w_idx,l]-cNa_tensor[w_idx-1,l])
        y2 = cNa_tensor[w_idx-1,l+1]+y2*(cNa_tensor[w_idx,l+1]-cNa_tensor[w_idx-1,l+1])
        sodium = z1*p["p(Na)/p(H')"]*torch.pow(10,y*(y1-y2)-y1-10.)
        sodium = torch.where((~flag_142) | (~flag_61),0,sodium)
    if debug:
        kappa_dict["kappa(Na)"] = sodium
    del flag_61

    # ======================================================
    # Mg
    # ======================================================
    flag_62 = (wavs_cm>wgrid_Mg[0]) & (wavs_cm<wgrid_Mg[-1])
    if torch.all((~flag_62) | (~flag_142)):
        magnesium = torch.zeros_like(carbon)
    else:
        fMg_tensor = torch.tensor(fgrid_Mg, dtype=T.dtype, device=T.device)
        y0 = lagrange_interp(
            T,
            fMg_tensor[T_idx-1],fMg_tensor[T_idx],fMg_tensor[T_idx+1],
            TC_tensor[T_idx-1],TC_tensor[T_idx],TC_tensor[T_idx+1]
        )
        y = torch.where(T_idx<=2,(y0-fMg_tensor[T_idx-1])/(fMg_tensor[T_idx]-fMg_tensor[T_idx-1]),y0)
        y = torch.where(T_idx>2, (y0-fMg_tensor[2])/2.992,y)
        y = torch.where(T_idx>4, (y0-fMg_tensor[4])/1.842,y)
        y = torch.where(T_idx>6, (y0-fMg_tensor[6])/2.556,y)
        y = torch.where(T_idx>11,(y0-fMg_tensor[11])/1.362,y)
        y = torch.where(T_idx>16,(y0-fMg_tensor[16])/1.496,y)
        l = torch.clamp(T_idx-1,max=1)
        l = torch.where(T_idx>2,2,l)
        l = torch.where(T_idx>4,3,l)
        l = torch.where(T_idx>6,4,l)
        l = torch.where(T_idx>11,5,l)
        l = torch.where(T_idx>16,6,l)
        w_idx = torch.ones_like(wavs_cm,dtype=torch.int32)
        for j in range(1,len(wgrid_Mg)):
            w_idx = torch.where(wavs_cm>wgrid_Mg[j],j+1,w_idx)
        wMg_tensor = torch.tensor(wgrid_Mg, dtype=T.dtype, device=T.device)
        cMg_tensor = torch.from_numpy(cmesh_Mg).to(device=T.device,dtype=T.dtype)
        y2 = (wavs_cm-wMg_tensor[w_idx-1])/(wMg_tensor[w_idx]-wMg_tensor[w_idx-1])
        y1 = cMg_tensor[w_idx-1,l]+y2*(cMg_tensor[w_idx,l]-cMg_tensor[w_idx-1,l])
        y2 = cMg_tensor[w_idx-1,l+1]+y2*(cMg_tensor[w_idx,l+1]-cMg_tensor[w_idx-1,l+1])
        magnesium = z1*p["p(Mg)/p(H')"]*torch.pow(10,y*(y1-y2)-y1-20.)
        magnesium = torch.where((~flag_142) | (~flag_62),0,magnesium)
    if debug:
        kappa_dict["kappa(Mg)"] = magnesium
    del flag_62

    # ======================================================
    # Continuum absorption coefficient
    # ======================================================
    kappac = (
        hminus+helmin+clmin+hneutr+h2min+h2plus+heneut+
        scatt1+scatt2+scatt3+escatt+
        carbon+sodium+magnesium
    )
    if debug:
        kappa_dict["kappa(cont)"]=kappac
        return kappa_dict
    return kappac

def selective_absorption(T,Pe,dopplerV,atom_i,atomic_properties):
    """
    kappa_L/N(H')
    """
    theta  = 5040/T
    log_gf = atom_i['log_gf']
    natom  = atom_i['idx']
    abu    = atomic_properties.abu[natom-1]
    l0     = atom_i['lambda0'] # Angstrom
    WN_l   = atom_i['WN_l'] # NIST cm^-1
    E_low  = const.c*const.h*WN_l*1e2/const.e # eV
    # print(log_gf)
    if log_gf:
        gf = 10**log_gf
    elif atom_i['f']:
        f = atom_i['f']
        ju = atom_i['ju']
        g = 2*ju+1
        gf = g*f
    else:
        return torch.zeros_like(T)
    eta00 = 1.49736e-2*gf*abu*(l0*1e-8)
    u1,u2,u3 = u123(natom,T)
    u12 = Saha(theta, atom_i['chi1'], u1, u2, Pe)
    u23 = Saha(theta, atom_i['chi2'], u2, u3, Pe)
    u33 = 1+u12*(1+u23) # (n1+n2+n3)/n1
    kappaL = eta00*torch.pow(10,-theta*E_low)/(u1*dopplerV*u33)
    kappaL = kappaL*(1.-torch.exp(-1.4388/(T*l0*1e-8)))
    return kappaL

def um(m,T):
    ret = 1.568399e+5/(T*m**2)
    return ret

def gff(T,x):
    ref=1.0828+3.865e-6*T+x*(7.564e-7+(4.92e-10-2.482e-15*T)*T+x*(5.326e-12+(-3.904e-15+1.879e-20*T)*T))
    return ref

def fkny(i,x):
    f1 = [14.47,-169.385,11.65,26.57,31.059,35.31,35.487,5.51,10.36,21.41,37.,25.54]
    f2 = [-2.,21.035,-1.91,-2.9,-3.3,-3.5,-3.6,-1.54,-1.86,-2.6,-3.69,-2.89]
    ret = f1[i-1]+x*f2[i-1]
    if i==2:
        ret = ret-0.727*x**2
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
