from .needs import *
from .math import *
from .network import *
from .phys import *
# from functools import deprecated
from ._compat import deprecated
from .utils import load_lines
import pkg_resources

@deprecated("Please use `synthesis2` instead")
class synthesis():
    """
    Forward modeling for synthesis Stokes Vector via stratified atomosphere
    """
    def __init__(
        self,
        abundances_csv = None, # Photospheric_Abundances.csv
        opacity_type = 'ATLAS',
        model_atmosphere = 'Quiet_sun',
        atoms_config = './atoms.json',
        user_option = dict(),
        **kwargs
    ):
        # self.VoigtNet = torch.load('./VoigtNet/voigtnet_new.pkl', weights_only=False)
        # self.VoigtFaradayNet = torch.load('./VoigtFaradayNet/voigtfaraday.pkl', weights_only=False)
        # df_read = pd.read_csv(abundances_csv)
        self.VoigtNet = torch.load(
            pkg_resources.resource_filename('pyPRT','data/voigtnet.pkl'),
            weights_only=False
        )
        self.VoigtFaradayNet = torch.load(
            pkg_resources.resource_filename('phPRT','data/voigtfaraday.pkl'),
            weights_only=False
        )
        df_read = atomic_config(abundances_csv).df
        self.Ar = np.power(10,np.array(df_read["Abundance (12+log[x]/[H])"].tolist())-12) # relative abundance
        self.Mr = np.array(df_read['Relative Mass'].tolist()) # relative mass g/mol
        temp = self.Ar*self.Mr
        self.X = 1/temp.sum()
        self.Y = self.X*temp[1]
        self.Z = 1-self.X-self.Y
        self.mu = temp.sum()/self.Ar.sum() # averavge molecule weights
        # self.mu = self.Ar.sum()/(self.Ar/self.Mr).sum()
        self.opacity_type = opacity_type
        self._OpacityInterpolator(opacity_type,**kwargs)
        self._InitAtmosphere(model_atmosphere, user_option)
        # self._Atoms(atoms_config)
        self.atoms = load_lines(lines)
        self.NumberDensity = dict()
        self.Nb = None
        self.Nt = None
        self.Nw = None
        self.check_prob = dict()
        self.hc_mu = np.cos(kwargs.get('heliocentric_angle',0.0)) # cosine of heliocentric angle

    def __call__(self, ltau, T, B, th, ph, vLos, vmic, vmac, lambdas, Pg=None,**kwargs):
        """
        Parameters:
        -----------
            ltau: Tensor (Nt,)
                log10(tau), opacity discrete grid
            T   : Tensor (Nb,Nt)
                T(tau), temperature at each ltau
            B, th, ph, vLos, vmic: Tensor (Nb,Bt)
                Similar as T, but for magnetic field magnitude, magnetic inclination, 
                magnetic azimuth, velocity of LOS, microturbulence velocity
            vmac: Tensor (Nb,)
                Macroturbulence velocity, does not rely on the opacity
            lambdas: Tensor (Nw,)
                Wavelength samplings  of synthetic spectra
            Pg  : Tensor (Nb,Nt)
                P(tau), gas pressure with opacity tau.
                If donnot provide, calculating via HSE.
                
            ! Nb,Nt,Nw means the batch size, tau sampling numbers and wavelength sampling numbers.

        Return:
        -----------
        Stokes: Tensor (Nb,Nw,4)
            Stokes vector (I,Q,U,V) at the given wavelength points.
        """
        rte_method = kwargs.get('rte_method','Hermitian')
        ltau = to_tensor(ltau,**kwargs)[None,:,None]
        T    = to_tensor(T   ,**kwargs)[:,:,None]
        B    = to_tensor(B   ,**kwargs)[:,:,None]
        th   = to_tensor(th  ,**kwargs)[:,:,None]
        ph   = to_tensor(ph  ,**kwargs)[:,:,None]
        vLos = to_tensor(vLos,**kwargs)[:,:,None]
        vmic = to_tensor(vmic,**kwargs)[:,:,None]
        vmac = to_tensor(vmac,**kwargs)[:,None]
        lambdas = to_tensor(lambdas,**kwargs)[None,None,:]
        self.Nb = T.size(0)
        self.Nt = ltau.size(1)
        self.Nw = lambdas.size(2)
        if Pg is None:
            Pg_top = self.Pg_interp(ltau.min().item())
            Pg_top = torch.tensor(Pg_top).unsqueeze(0).unsqueeze(0)
            Pg = self._HSE_solver(ltau,T,vmic,B,Pg_top,**kwargs)
        else:
            Pg = to_tensor(Pg,**kwargs)[:,:,None]
        self._NumberDensity(T,Pg,**kwargs)
        K_matx = self.Propagation_matrix(Pg,T,B,th,ph,vLos,vmic,lambdas,**kwargs)
        # O_oprt = self.Evolution_operator(K_matx, log_tau, lambdas, **kwargs)
        S_func = self.Source_function(T, lambdas,**kwargs)
        # Stokes = self.Intensity_integration(O_oprt, K_matx, S_func, log_tau, **kwargs)
        # Stokes = Hermition_Method(ltau,K_matx,S_func)
        if rte_method=='RK4':
            Stokes = RK4_method(ltau,K_matx,S_func)
        elif rte_method=='Hermitian':
            Stokes = Hermitian_method(ltau,K_matx,S_func)
        elif rte_method=='DELO':
            Stokes = DELO_method(ltau,K_matx,S_func)
        else:
            raise ValueError(f"{rte_method} is not the supported method: ('Rk4', 'Hermitian')")
        Stokes_obs = Apply_Macroturbulence(Stokes,vmac, lambdas)
        return Stokes

    def _OpacityInterpolator(self, opacity_type,**kwargs):
        if opacity_type=='Opacity Project':
            # tab_summs = kwargs.get('OPtab_summs', './table_summaries.csv')
            tab_summs = pkg_resources.resource_filename(
                'pyprt',
                'data/OPtab_summaries.csv'
            )
            tab_summs = pd.read_csv(tab_summs)
            XYZ = tab_summs.values[:,1:4]
            OPtab_idx = np.argmin(np.abs(XYZ-np.array([self.X,self.Y,self.Z])[None,:]).sum(axis=1))
            # OPtabs_path = kwargs.get('OP_tab', './OPtabs/')
            OPtabs_path = pkg_resources.resource_filename('pyPRT', 'data/OPtabs')
            OPtabs_file = sorted(glob.glob(os.path.join(OPtabs_path,'*.csv')))[OPtab_idx]
            df_read = pd.read_csv(OPtabs_file)
            logRlist = np.arange(-8.0,1.0+0.5,0.5)
            logTlist = df_read.values[:,0]
            opacitys  = df_read.values[:,1:]
            grid_logR = torch.from_numpy(logRlist)
            grid_logT = torch.from_numpy(logTlist)
            OP_tensor = torch.from_numpy(opacitys)
            OP_interp = TensorGridInterpolator((grid_logT,grid_logR),OP_tensor)
            self.kapp_interp = OP_interp
        elif opacity_type=='ATLAS':
            atlas_tab = pkg_resources.resource_filename(
                'pyprt',
                'data/ATLAS_solar_ODF.txt'
            )
            atlas_odf = np.loadtxt(atlas_tab, skiprows=2)
            atlas_logT = np.unique(atlas_odf[:,0])
            atlas_logP = np.unique(atlas_odf[:,1])
            atlas_vmic = np.array([0,1,2,4,8.])
            nT = len(atlas_logT)
            nP = len(atlas_logP)
            nV = len(atlas_vmic)
            atlas_kapp = atlas_odf[:,2:7].reshape(nT,nP,nV)
            atlas_kapp_tensor = torch.from_numpy(atlas_kapp)
            atlas_logT_tensor = torch.from_numpy(atlas_logT)
            atlas_logP_tensor = torch.from_numpy(atlas_logP)
            atlas_vmic_tensor = torch.from_numpy(atlas_vmic)
            atlas_kapp_interp = TensorGridInterpolator(
                (atlas_logT_tensor,atlas_logP_tensor,atlas_vmic_tensor),
                atlas_kapp_tensor
            )
            self.kapp_interp = atlas_kapp_interp
        else:
            raise ValueError(f'`opacity_type` only support "ATLAS" or "Opacity Project", but get {opacity_type}')

    def _InitAtmosphere(self, model_atmosphere, user_option):
        atmos_file = pkg_resources.resource_filename('pyPRT','data/atomsphere.pkl')
        with open(atmos_file, 'rb') as f:
            self.atmosphere = pickle.load(f)
        # model_path = self.atmosphere['initial_guess']['path']
        model_path  = pkg_resources.resource_filename('pyPRT','data/model_atmosphere')
        model_files = sorted(glob.glob(os.path.join(model_path,'*.txt')))
        model_names = [os.path.splitext(os.path.basename(ifile))[0] for ifile in model_files]
        if model_atmosphere in model_names:
            self.atmosphere['initial_guess']['choice']=model_atmosphere
            self.atmosphere['initial_guess']['file']=os.path.join(model_path,model_atmosphere+'.txt')
        elif model_atmosphere=='User':
            self.atmosphere['initial_guess']['choice']=model_atmosphere
            try:
                self.atmosphere['initial_guess']['file']=user_option['initial_atmosphere_txt']
            except KeyError as e:
                raise KeyError(f"`user_option` loss the necessary key: {e} when `model_atmosphere`='User'")
        else:
            raise FileNotFoundError(
                f"{model_atomsphere} is not included in {model_names} or 'User'"
            )
        if self.atmosphere['initial_guess']['file']:
            data = np.loadtxt(self.atmosphere['initial_guess']['file'],skiprows=2)
            ltau = data[:,0]
            Pg   = data[:,-2]
            if ltau[0]>ltau[-1]:
                ltau = ltau[::-1]
                Pg   = Pg[::-1]
            self.Pg_interp = CubicSpline(ltau, Pg)

    def _Atoms(self,atoms_config):
        with open(atoms_config, 'r') as f:
            self.atoms = json.load(f)

    def _init_size(self,data,lambdas):
        self.Nb = data.size(0)
        self.Nt = data.size(1)
        self.Nw = lambdas.size(2)

    def _HSE_solver(self, ltau, T, vmic, B, Pg_top,**kwargs):
        """
        Computing the gas pressure based on the HSE condition:
        dP/dtau = -g/kapp(tau)
        kapp(tau) = kapp(T(tau),P(tau))
        Pg(tau) = Pg_top+NIntergration[-g/kapp(t),{t,tau_min,tau_max}]
        """
        # R = const.R*1e7
        g = 2.74e4*self.hc_mu
        Pmag = B**2/(8*np.pi)
        dPmag = torch.diff(Pmag,dim=1,n=2,prepend=Pmag[:,:1],append=Pmag[:,-1:])
        def rfunc(ltaui, Pi):
            iltau = torch.searchsorted(ltau.squeeze(), ltaui)
            Ti    = T[:,iltau]
            vmici = vmic[:,iltau]
            # print(f"dPmag size: {dPmag.size()}")
            dPmagi = dPmag[:,iltau]
            if self.opacity_type=='ATLAS':
                coords = torch.stack([torch.log10(Ti),torch.log10(Pi),vmici],dim=-1)
                lkapp = self.kapp_interp(coords)
            elif self.opacity_type=='Opacity Project':
                T6 = Ti*1e-6
                # rhoi = Pi/(R*Ti)
                logR = torch.log10(rho/T6**3)
                logT = torch.log10(Ti)
                coords = torch.stack([logT,logR],dim=-1)
                lkapp = self.kapp_interp(coords)
            # print(iltau.item(), ltaui.item(), lkapp.item(), ltaui.item(), Pi.item())
            ret = np.log(10)*10**ltaui*g/torch.pow(10,lkapp)-dPmagi
            # print(ret)
            return ret
        Pg = RK4_solver(rfunc)(ltau.squeeze(), Pg_top)
        Pg = torch.stack(Pg,dim=1)
        if kwargs.get('check_Pg',None):
            self.check_prob['Pg']=Pg
        return Pg

    def _NumberDensity(self, T, Pg, **kwargs):
        """
        Calculating the necessary number density via ionization equilibrium (Saha equation),
        and chemical equilbrium (charge conservation & dissociation equilibrium)
        """
        me = const.m_e*1e3
        mp = const.m_p*1e3
        kB = const.k*1e7
        h  = const.h*1e7
        e0 = const.e*1e7
        Ng = Pg/(kB*T)
        NH = Ng/(self.Ar.sum())
        atoms = self.atmosphere['components']['atoms']
        molecules = self.atmosphere['components']['molecules']
        # partition function
        g_H = atoms['H']['gi']
        g_Hp = atoms['Hp']['gi']
        g_Hm = atoms['Hm']['gi']
        g_H2p = molecules['H2p']['gi']
        T_vib = molecules['H2p']['T_vib']
        T_rot = molecules['H2p']['T_rot']
        g_H2p = g_H2p**torch.exp(-T_vib/(2*T))/(1-torch.exp(-T_vib/T))*T/T_rot
        ai = molecules['H2']['Poly_partition_function']
        g_H2 = torch.stack([ai[i]*T**i for i in range(len(ai))],dim=0).sum(dim=0)  
        # ionization potential
        IP_H = atoms['H']['ionization_potential']*e0
        IP_H2 = molecules['H2']['ionization_potential']*e0
        D_H2 = molecules['H2']['diss_energy']*e0
        IP_Hm = atoms['Hm']['ionization_potential']*e0
        IP_H2 = molecules['H2']['ionization_potential']*e0
        # coefficient
        coef1 = (2*np.pi*me*kB*T/h**2)**1.5
        # Saha equation
        S_H = coef1*2*g_Hp/g_H*torch.exp(-IP_H/(kB*T))
        S_Hm = coef1*2*g_H/g_Hm*torch.exp(-IP_Hm/(kB*T))
        S_H2 = coef1*2*g_H2p/g_H2*torch.exp(-IP_H2/(kB*T))
        Kn = coef1*(0.5*mp/me)**1.5*g_H**2/g_H2*torch.exp(-D_H2/(kB*T))
        # Numerical solving
        a = S_H/NH
        b = S_Hm/NH
        c = Kn/NH
        d = S_H2/NH
        def Func(x):
            y = torch.sqrt((a*x+d/c*x**2)/(1+x/b))
            F = 2*(y+d)/c*x**2+(a+y+y**2/b)*x-y
            return F
        x0   = torch.ones_like(NH)*0.90
        x1   = torch.ones_like(NH)
        xsol = Newton_solver(Func)(x0,x1)
        # print(xsol.squeeze())
        N_H  = xsol*NH
        N_e  = torch.sqrt((a*xsol+d/c*xsol**2)/(1+xsol/b))*NH
        N_Hp = S_H/N_e*N_H
        N_Hm = N_e/S_Hm*N_H
        N_H2 = N_H**2/Kn
        N_H2p= S_H2/Kn*N_H**2/N_e
        N_He = NH*self.Ar[1]
        self.N = dict(
            H = N_H,
            e = N_e,
            Hp = N_Hp,
            Hm = N_Hm,
            H2 = N_H2,
            H2p = N_H2p,
            gas = Ng,
            Htotal = NH,
            He = N_He
        )


    def Propagation_matrix(
        self,
        Pg,T,B,th,ph,vLos,vmic,lambdas,
        **kwargs
    ):
        """
        Calculating the propagation matrix by given parameters
        """
        eta_b,eta_p,eta_r,rho_b,rho_p,rho_r = self.AbsorptionDispersion_profile(
            Pg,T,B,th,ph,vLos,vmic,lambdas,
            **kwargs
        )
        etaI = 1+0.5*(eta_p*torch.sin(th)**2+0.5*(eta_r+eta_b)*(1+torch.cos(th)**2))
        etaQ = 0.5*(eta_p-0.5*(eta_r+eta_b))*torch.sin(th)**2*torch.cos(2*ph)
        etaU = 0.5*(eta_p-0.5*(eta_r+eta_b))*torch.sin(th)**2*torch.sin(2*ph)
        etaV = 0.5*(eta_r-eta_b)*torch.cos(th)
        rhoQ = 0.5*(rho_p-0.5*(rho_b+rho_r))*torch.sin(th)**2*torch.cos(2*ph)
        rhoU = 0.5*(rho_p-0.5*(rho_b+rho_r))*torch.sin(th)**2*torch.sin(2*ph)
        rhoV = 0.5*(rho_r-rho_b)*torch.cos(th)
        K_matx = torch.stack([
           torch.stack([ etaI, etaQ, etaU, etaV],dim=-1),
           torch.stack([ etaQ, etaI, rhoV,-rhoU],dim=-1),
           torch.stack([ etaU,-rhoV, etaI, rhoQ],dim=-1),
           torch.stack([ etaV, rhoU,-rhoQ, etaI],dim=-1)
        ],dim=-2)
        return K_matx

    def AbsorptionDispersion_profile(
        self,
        Pg,T,B,th,ph,vLos,vmic,lambdas,
        **kwargs):
        """
        Calculating the absorption and dispersion profile for the pi-and sigma components
        """
        device = T.device
        # VoigtNet = torch.load('./VoigtNet/voigtnet_new.pkl',weights_only=False).to(device=device,dtype=T.dtype)
        # VoigtFaradayNet = torch.load('./VoigtFaradayNet/voigtfaraday.pkl',weights_only=False).to(device=device,dtype=T.dtype)
        VoigtNet = self.VoigtNet.to(device=device,dtype=T.dtype)
        VoigtFaradayNet = self.VoigtFaradayNet.to(device=device,dtype=T.dtype)
        size0 = T.shape
        c = const.c # m/s
        R = const.R
        if self.Nb is None:
            self._init_size(T,lambdas)

        first=True
        for iatom in self.atoms:
            atom_i = self.atoms[iatom]
            jl = atom_i['jl']
            ju = atom_i['ju']
            lambda0 = atom_i['lambda0']
            M = atom_i['M'] # g/mol
            gl = atom_i['Landel']
            gu = atom_i['Landeu']
            Sb,Sp,Sr,m2b,m2p,m2r = Unnormalized_strengths(jl,ju)
            Sb = Sb/Sb.sum()
            Sp = Sp/Sp.sum()
            Sr = Sr/Sr.sum()
            dopplerV = torch.sqrt(2*R*T/M*1e3+(vmic*1e3)**2) # m/s
            dopplerW = lambda0/c*dopplerV # Angstrom
            lambdaB = 4.67e-13*lambda0**2*B
            u0 = (lambdas-lambda0)/dopplerW
            uB0 = lambdaB/dopplerW
            Gamma = self.DampingParameter(atom_i, T)
            a = (Gamma*(lambda0)**2/(4*np.pi*c*1e10*dopplerW))
            a = a.expand(-1,-1,self.Nw).flatten()
            self.check_prob['a'] = a.detach().cpu().numpy()
            self.check_prob['dopplerW'] = dopplerW.detach().cpu().numpy()
            eta0 = self.Line_to_continuum_ratio(
                Pg,T,vmic,dopplerW,atom_i,
                **kwargs)
            for alpha in [1,0,-1]:
                if alpha==1:
                    m2,S = m2b,Sb
                    m1 = m2-1
                elif alpha==0:
                    m2,S = m2p,Sp
                    m1 = m2
                else:
                    m2,S = m2r,Sr
                    m1 = m2+1
                for i, (Si,ml,mu) in enumerate(zip(S,m1,m2)):
                    uLos = (lambda0-lambdaB*(ml*gl-mu*gu))/dopplerW*vLos*1e3/c
                    # uLos = (lambda0)/dopplerW*vLos*1e3/c
                    uB = uB0*(ml*gl-mu*gu)
                    u = (u0+uB-uLos).flatten()
                    VoigtNet.eval()
                    VoigtFaradayNet.eval()
                    if i==0:
                        eta_alpha = Si*torch.exp(VoigtNet(u,a))*np.pi**0.5
                        rho_alpha = Si*VoigtFaradayNet(u,a)*np.pi**0.5
                    else:
                        eta_alpha+= Si*torch.exp(VoigtNet(u,a))*np.pi**0.5
                        rho_alpha+= Si*VoigtFaradayNet(u,a)*np.pi**0.5
                eta_alpha = eta_alpha.reshape(self.Nb,self.Nt,self.Nw)
                rho_alpha = rho_alpha.reshape(self.Nb,self.Nt,self.Nw)
                if alpha==1:
                    ieta_b = eta_alpha*eta0
                    irho_b = rho_alpha*eta0
                elif alpha==0:
                    ieta_p = eta_alpha*eta0
                    irho_p = rho_alpha*eta0
                else:
                    ieta_r = eta_alpha*eta0
                    irho_r = rho_alpha*eta0
            if first:
                eta_b,eta_p,eta_r = ieta_b,ieta_p,ieta_r
                rho_b,rho_p,rho_r = irho_b,irho_p,irho_r
                first=False
            else:
                eta_b+=ieta_b
                eta_p+=ieta_p
                eta_r+=ieta_r
                rho_b+=irho_b
                rho_p+=irho_p
                rho_r+=irho_r
        return eta_b,eta_p,eta_r,rho_b,rho_p,rho_r

    def DampingParameter(self, atom_i, T):
        Y_rad = atom_i['Y_rad']
        Y_vdw = atom_i['Y_vdw']
        M     = atom_i['M']
        Ni    = [self.N['H'],self.N['H2'],self.N['He']]
        vdw_contributor = self.atmosphere['van_der_waals_contributor']
        for i,xi in enumerate(vdw_contributor):
            alphai = vdw_contributor[xi]['polarizability']
            Mi = vdw_contributor[xi]['mass']
            if i==0:
                alphaH = vdw_contributor[xi]['polarizability']
                MH = vdw_contributor[xi]['mass']
                Y_6 = Ni[i]*Y_vdw*(T*1e-4)**0.3
            else:
                Y_6 += Ni[i]*Y_vdw*(T*1e-4)**0.3*(alphai/alphaH)**0.4*((1/M+1/Mi)/(1/M+1/MH))**0.3
        Gamma = Y_rad+Y_6
        return Gamma

    def Line_to_continuum_ratio(
        self,
        Pg,T,vmic,dopplerW,atom_i,
        **kwargs):
        me = const.m_e*1e3
        c  = const.c*1e2
        e0 = const.e*c/10
        kB = const.k*1e7
        Rg = const.R*1e7
        h  = const.h*1e7
        idx = atom_i['idx']
        Ni = self.N['Htotal']*self.Ar[idx-1]
        log_gf = atom_i['log_gf']
        if log_gf:
            gf = 10**log_gf
        elif atom_i['f']:
            f = atom_i['f']
            ju = atom_i['ju']
            g = 2*ju+1
            gf = g*f
        else:
            return torch.zeros_like(T)
        l0 = atom_i['lambda0'] # Angstrom
        Mg = (self.Mr*self.Ar).sum()/(self.Ar.sum())
        # Polynomial partition function approximations of 344 atomic and molecular species.
        # Irwin, A.W. 1981
        u1 = polynomial_partition_function(atom_i['polyQ'] ,T)
        u2 = polynomial_partition_function(atom_i['polyQ1'],T)
        u3 = polynomial_partition_function(atom_i['polyQ2'],T)
        theta = const.e/(np.log(10)*const.k)/T
        Pe = self.N['e']*kB*T
        u12 = Saha(theta, atom_i['chi1'], u1, u2, Pe)
        u23 = Saha(theta, atom_i['chi2'], u2, u3, Pe)
        u33 = 1+u12*(1+u23) # (n1+n2+n3)/n1
        eta00 = 1.49736e-2*gf*self.Ar[idx-1]*(l0*1e-8)*self.N['Htotal']
        WN_l = atom_i['WN_l'] # NIST cm^-1
        E_low = const.c*const.h*WN_l*1e2/const.e # eV
        dopplerV = dopplerW/l0*c # cm/s
        rg = Pg/(T*Rg/self.mu) # gas density rho_g: P = rho R/mu T
        if kwargs.get('check_rg',None):
            self.check_prob['rg']=rg
        if self.opacity_type=='Opacity Project':
            Ng = self.N['gas']
            rho = Mg*Ng*kB/Rg
            logR = torch.log10(rho/(T*1e-6)**3)
            logT = torch.log10(T)
            points = torch.stack([logT,logR],dim=-1)
            kappa_C = torch.pow(10,self.kapp_interp(points))*rg
            # kappa_C = torch.pow(10,self.kapp_interp(points))
        elif self.opacity_type=='ATLAS':
            Ng = self.N['gas']
            Pg = Ng*kB*T
            logT = torch.log10(T)
            logP = torch.log10(Pg)
            points = torch.stack([logT,logP,vmic],dim=-1)
            kappa_C = torch.pow(10,self.kapp_interp(points))*rg
            # kappa_C = torch.pow(10,self.kapp_interp(points))
        kappa_L = eta00*torch.pow(10,-theta*E_low)/(u1*dopplerV*u33*kappa_C)
        kappa_L = kappa_L*(1.-torch.exp(-1.4388/(T*l0*1e-8)))        
        eta0 = kappa_L # kappa_L/kappa_C
        if kwargs.get('check_kappa',None):
            self.check_prob['kappa']=dict(kappa_C=kappa_C,kappa_L=kappa_L,eta0=eta0)
        return eta0

    def Evolution_operator(
        self,
        K_matx, ltau, lambdas,
        **kwargs
    ):
        pass

    def Source_function(self, T, lambdas, **kwargs):
        # nu = const.c/lambdas*1e10
        # SI = Planck_Bnu(T, nu)
        SI = Planck_Blambda(T, lambdas)
        SQ = torch.zeros_like(SI)
        SU = torch.zeros_like(SI)
        SV = torch.zeros_like(SI)
        S  = torch.stack([SI,SQ,SU,SV], axis=-1)
        return S