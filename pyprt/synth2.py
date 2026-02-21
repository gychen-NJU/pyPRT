from .needs import *
from .math import *
from .network import *
from .phys import *
from .atoms import atomic_config,atomic_properties
from .pressure import ie_pressure
from .absorption import continuum_absorption, selective_absorption
from .utils import load_lines,get_xyz
import pkg_resources

class synthesis2():
    """
    Forward modeling for synthesis Stokes Vector via stratified atomosphere

    Parameters:
    -------------
        table_abu: None  -> tables include abundance and atomic weight infomation
        solver   : str   -> solver for RTE, default is 'Hermitian'
        refidx   : float -> refractive index of the medium, default is 1.
        lines    : dict  -> lines for synthesis, default is {"FeI":["6301.5","6302.5"]}
    """
    def __init__(
        self,
        table_abu = None,
        solver = 'Hermitian',
        refidx = 1., # refractive index
        lines = {"FeI":["6301.5","6302.5"]},
        opacity_type='formula' # 'formula', 'ATLAS', 'Opacity Project'
        ):
        self.AP = atomic_config(table_abu)
        self.lines = load_lines(lines)
        self.rte_solver = RTE_solver(solver)
        self.device = None
        self.dtype = None
        self.refidx=refidx
        self.opacity_type=opacity_type
        self.van_der_waals_contributor = {
            'H': {'polarizability': 6.6e-25, 'mass': 1},
            'H2': {'polarizability': 8e-25, 'mass': 2},
            'He': {'polarizability': 2.1e-25, 'mass': 3.97}
            }
        self.debug_log = dict()
        self.RF = dict()
        self._OpacityInterpolator(opacity_type)

    def __call__(self,lambdas,ltau,T,Pe,B,th,ph,vLos,vmic,vmac,**kwargs):
        """
        Parameters:
        -----------
            lambdas: Tensor (Nw,)   | lambdas | wavelength                     | [Angstrom]
            ltau   : Tensor (Nt,)   | lg(tau) | lograrithmic optical thickness | []
            Pe     : Tensor (Nb,Nt) | P(e-)   | electron pressure with opacity | [dyn/cm^2]
            T      : Tensor (Nb,Nt) | T       | temperature at each ltau       | [K]
            B      : Tensor (Nb,Nt) | B       | magnetic field magnitude       | [G]
            th     : Tensor (Nb,Nt) | th      | magnetic inclination           | [rad]
            ph     : Tensor (Nb,Nt) | ph      | magnetic azimuth               | [rad]
            vLos   : Tensor (Nb,Nt) | vLos    | velocity of LOS                | [km/s]
            vmic   : Tensor (Nb,Nt) | vmic    | microturbulence velocity       | [km/s]
            vmac   : Tensor (Nb,)   | vmac    | macroturbulence velocity       | [km/s]
                
            ! Nb,Nt,Nw means the batch size, tau sampling numbers and wavelength sampling numbers.

        Return:
        -----------
        Stokes: Tensor (Nb,Nw,4)
            Stokes vector (I,Q,U,V) at the given wavelength points.
        """
        self.device = kwargs.get('device',T.device)
        self.dtype = kwargs.get('dtype',T.dtype)
        compute_RF = kwargs.get('compute_RF',False)
        self.Nb = T.size(0)
        self.Nt = ltau.size(0)
        self.Nw = lambdas.size(0)
        if compute_RF:
            # T    = T.requires_grad_(True)
            # Pe   = Pe.requires_grad_(True)
            # B    = B.requires_grad_(True)
            # th   = th.requires_grad_(True)
            # ph   = ph.requires_grad_(True)
            # vLos = vLos.requires_grad_(True)
            # vmic = vmic.requires_grad_(True)
            X = torch.cat([T,Pe,B,th,ph,vLos,vmic],dim=-1)
        t  = to_tensor(T   ,**kwargs)[:,:,None]
        p  = to_tensor(Pe  ,**kwargs)[:,:,None]
        b  = to_tensor(B   ,**kwargs)[:,:,None]
        g  = to_tensor(th  ,**kwargs)[:,:,None]
        f  = to_tensor(ph  ,**kwargs)[:,:,None]
        v  = to_tensor(vLos,**kwargs)[:,:,None]
        m = to_tensor(vmic,**kwargs)[:,:,None]
        vmac = to_tensor(vmac,**kwargs)[:,None]
        ltau = to_tensor(ltau,**kwargs)[None,:,None]
        lambdas = to_tensor(lambdas,**kwargs)[None,None,:]
        theta = 5040/t
        P = ie_pressure(theta,p,atomic_properties=self.AP)
        K_matx = self._Propagation_matrix(P,t,p,b,g,f,v,m,lambdas,**kwargs)
        S_func = self._Source_function(t, lambdas,**kwargs)
        Stokes = self.rte_solver(ltau,K_matx,S_func)
        # print(Stokes.shape)
        Stokes_obs = Apply_Macroturbulence_Conv(Stokes,vmac, lambdas)
        if compute_RF:
            # self._ResponseFunction(Stokes_obs,T,Pe,B,th,ph,vLos,vmic)
            self._ResponseFunction_fast(lambdas[0,0],ltau[0,:,0],X,vmac,**kwargs)
        return Stokes_obs

    def _ResponseFunction_fast(self,lambdas,ltau,X,vmac,**kwargs):
        print(" ### Start computing Response Function ###")
        Nb,Nt,Nw=self.Nb,self.Nt,self.Nw
        S_str = ["I","Q","U","V"]
        param_str = ['T','Pe','B','gamma','phi','vLos','vmic']
        RF = dict()
        for iS in S_str:
            RF[iS]={iP:torch.zeros((Nb,Nt,Nw),device=self.device,dtype=self.dtype) for iP in param_str}
        for ib,(xi,maci) in enumerate(zip(X,vmac)):
            def sfunc(xi):
                ti,pi,bi,gi,fi,vi,mi = xi.reshape(7,-1)
                ti = ti[None,:]
                pi = pi[None,:]
                bi = bi[None,:]
                gi = gi[None,:]
                fi = fi[None,:]
                vi = vi[None,:]
                mi = mi[None,:]
                # print(maci.shape,vmac.shape)
                si = self.__call__(lambdas,ltau,ti,pi,bi,gi,fi,vi,mi,maci)
                return si[0]
            xi.requires_grad_(True)
            jacobi = torch.autograd.functional.jacobian(sfunc,xi).detach()
            jacobi = jacobi.permute(2,0,1) # (Nt*7,Nw,4)
            if (debug:=kwargs.get('debug',None)):
                if debug.get('check_jacobi',None):
                    print('recording jacobi')
                    self.debug_log['jacobi']=jacobi.detach().cpu().clone()
            for ip,iP in enumerate(param_str):
                RF['I'][iP][ib]=jacobi[ip*Nt:(ip+1)*Nt,:,0]
                RF['Q'][iP][ib]=jacobi[ip*Nt:(ip+1)*Nt,:,1]
                RF['U'][iP][ib]=jacobi[ip*Nt:(ip+1)*Nt,:,2]
                RF['V'][iP][ib]=jacobi[ip*Nt:(ip+1)*Nt,:,3]
        self.RF=RF

    def _ResponseFunction(
        self,
        Stokes_obs,T,Pe,B,th,ph,vLos,vmic,
        **kwargs
    ):
        """
        Calculating the response function by given parameters
        """
        print(" ### Start computing Response Function ###")
        Nb,Nt,Nw=self.Nb,self.Nt,self.Nw
        S_str = ["I","Q","U","V"]
        param_str = ['T','Pe','B','gamma','phi','vLos','vmic']
        RF = dict()
        for iS in S_str:
            RF[iS]={iP:torch.zeros((Nb,Nt,Nw),device=self.device,dtype=self.dtype) for iP in param_str}
        for ib,s in enumerate(Stokes_obs):
            i,q,u,v = s.T
            for iw in range(s.size(0)):
                # print(f" ## Traverse the wavelength point: {iw+1:4d}/{Nw}")
                RF['I']['T'][ib,:,iw]=self._grad(self,i[iw],T).detach().cpu()[ib]
                RF['Q']['T'][ib,:,iw]=self._grad(self,q[iw],T).detach().cpu()[ib]
                RF['U']['T'][ib,:,iw]=self._grad(self,u[iw],T).detach().cpu()[ib]
                RF['V']['T'][ib,:,iw]=self._grad(self,v[iw],T).detach().cpu()[ib]
                RF['I']['Pe'][ib,:,iw]=self._grad(self,i[iw],Pe).detach().cpu()[ib]
                RF['Q']['Pe'][ib,:,iw]=self._grad(self,q[iw],Pe).detach().cpu()[ib]
                RF['U']['Pe'][ib,:,iw]=self._grad(self,u[iw],Pe).detach().cpu()[ib]
                RF['V']['Pe'][ib,:,iw]=self._grad(self,v[iw],Pe).detach().cpu()[ib]
                RF['I']['B'][ib,:,iw]=self._grad(self,i[iw],B).detach().cpu()[ib]
                RF['Q']['B'][ib,:,iw]=self._grad(self,q[iw],B).detach().cpu()[ib]
                RF['U']['B'][ib,:,iw]=self._grad(self,u[iw],B).detach().cpu()[ib]
                RF['V']['B'][ib,:,iw]=self._grad(self,v[iw],B).detach().cpu()[ib]
                RF['I']['gamma'][ib,:,iw]=self._grad(self,i[iw],th).detach().cpu()[ib]
                RF['Q']['gamma'][ib,:,iw]=self._grad(self,q[iw],th).detach().cpu()[ib]
                RF['U']['gamma'][ib,:,iw]=self._grad(self,u[iw],th).detach().cpu()[ib]
                RF['V']['gamma'][ib,:,iw]=self._grad(self,v[iw],th).detach().cpu()[ib]
                RF['I']['phi'][ib,:,iw]=self._grad(self,i[iw],ph).detach().cpu()[ib]
                RF['Q']['phi'][ib,:,iw]=self._grad(self,q[iw],ph).detach().cpu()[ib]
                RF['U']['phi'][ib,:,iw]=self._grad(self,u[iw],ph).detach().cpu()[ib]
                RF['V']['phi'][ib,:,iw]=self._grad(self,v[iw],ph).detach().cpu()[ib]
                RF['I']['vLos'][ib,:,iw]=self._grad(self,i[iw],vLos).detach().cpu()[ib]
                RF['Q']['vLos'][ib,:,iw]=self._grad(self,q[iw],vLos).detach().cpu()[ib]
                RF['U']['vLos'][ib,:,iw]=self._grad(self,u[iw],vLos).detach().cpu()[ib]
                RF['V']['vLos'][ib,:,iw]=self._grad(self,v[iw],vLos).detach().cpu()[ib]
                RF['I']['vmic'][ib,:,iw]=self._grad(self,i[iw],vmic).detach().cpu()[ib]
                RF['Q']['vmic'][ib,:,iw]=self._grad(self,q[iw],vmic).detach().cpu()[ib]
                RF['U']['vmic'][ib,:,iw]=self._grad(self,u[iw],vmic).detach().cpu()[ib]
                RF['V']['vmic'][ib,:,iw]=self._grad(self,v[iw],vmic).detach().cpu()[ib]
        self.RF = RF

    @staticmethod
    def _grad(self,y,x):
        g = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
        return g

    def _Propagation_matrix(
        self,
        p,T,Pe,B,th,ph,vLos,vmic,lambdas,
        **kwargs
    ):
        """
        Calculating the propagation matrix by given parameters
        """
        eta_b,eta_p,eta_r,rho_b,rho_p,rho_r = self._AbsorptionDispersion_profile(
            p,T,Pe,B,vLos,vmic,lambdas,
            **kwargs
        )
        if self.opacity_type=='formula':
            wref = to_tensor(np.array([5000]),**kwargs)[None,None,:].to(device=self.device,dtype=self.dtype)
            kappa5 = continuum_absorption(wref,   T,Pe,p,refidx=self.refidx)
            kappaC = continuum_absorption(lambdas,T,Pe,p,refidx=self.refidx)
        elif self.opacity_type=='ATLAS':
            Rg = const.R*1e7
            Pg = p['pg']
            kB = const.k*1e7
            NH = p["p(H')"]/(kB*T)
            amw= self.AP.amw.sum()/self.AP.abu.sum()
            rg = Pg/(T*Rg/amw)
            logT = torch.log10(T)
            logP = torch.log10(Pg)
            points = torch.stack([logT,logP,vmic],dim=-1)
            kappaC = torch.pow(10,self.kapp_interp(points))*rg/NH
            kappa5 = kappaC
        elif self.opacity_type=='Opacity Project':
            Rg = const.R*1e7
            kB = const.k*1e7
            Pg = p['pg']
            NH = p["p(H')"]/(kB*T)
            amw= self.AP.amw.sum()/self.AP.abu.sum()
            rg = Pg/(T*Rg/amw)
            logR = torch.log10(rg/(T*1e-6)**3)
            logT = torch.log10(T)
            points = torch.stack([logT,logR],dim=-1)
            kappaC = torch.pow(10,self.kapp_interp(points))*rg/NH
            kappa5 = kappaC
        else:
            raise ValueError(f'`opacity_type` only support "formula","ATLAS", or "Opacity Project", but get {self.opacity_type}')
        if (debug := kwargs.get('debug',None)):
            if debug.get('check_kappa'):
                self.debug_log['kappa5']=kappa5
                self.debug_log['kappaC']=kappaC
        etaI = kappaC/kappa5+0.5*(eta_p*torch.sin(th)**2+0.5*(eta_r+eta_b)*(1+torch.cos(th)**2))/kappa5
        etaQ = 0.5*(eta_p-0.5*(eta_r+eta_b))*torch.sin(th)**2*torch.cos(2*ph)/kappa5
        etaU = 0.5*(eta_p-0.5*(eta_r+eta_b))*torch.sin(th)**2*torch.sin(2*ph)/kappa5
        etaV = 0.5*(eta_r-eta_b)*torch.cos(th)/kappa5
        rhoQ = 0.5*(rho_p-0.5*(rho_b+rho_r))*torch.sin(th)**2*torch.cos(2*ph)/kappa5
        rhoU = 0.5*(rho_p-0.5*(rho_b+rho_r))*torch.sin(th)**2*torch.sin(2*ph)/kappa5
        rhoV = 0.5*(rho_r-rho_b)*torch.cos(th)/kappa5
        K_matx = torch.stack([
           torch.stack([ etaI, etaQ, etaU, etaV],dim=-1),
           torch.stack([ etaQ, etaI, rhoV,-rhoU],dim=-1),
           torch.stack([ etaU,-rhoV, etaI, rhoQ],dim=-1),
           torch.stack([ etaV, rhoU,-rhoQ, etaI],dim=-1)
        ],dim=-2)
        if (debug := kwargs.get('debug',None)):
            if debug.get('check_Kmat',None):
                self.debug_log['Kmat'] = dict(
                    etaI=etaI,etaQ=etaQ,etaU=etaU,etaV=etaV,rhoQ=rhoQ,rhoU=rhoU,rhoV=rhoV,
                    K_matx=K_matx,
                )
        return K_matx

    def _AbsorptionDispersion_profile(
        self,
        p,T,Pe,B,vLos,vmic,lambdas,
        **kwargs):
        """
        Calculating the absorption and dispersion profile for the pi-and sigma components
        eta_{l,r,p}*kappaL, rho_{l,r,p}*kappaL
        """
        c = const.c # m/s
        R = const.R

        first=True
        for iline in self.lines:
            line_i = self.lines[iline]
            jl = line_i['jl']
            ju = line_i['ju']
            lambda0 = line_i['lambda0'] # Angstrom
            M = line_i['M'] # g/mol
            gl = line_i['Landel']
            gu = line_i['Landeu']
            Sb,Sp,Sr,m2b,m2p,m2r = Unnormalized_strengths(jl,ju)
            Sb = Sb/Sb.sum()
            Sp = Sp/Sp.sum()
            Sr = Sr/Sr.sum()
            dopplerV = torch.sqrt(2*R*T/M*1e3+(vmic*1e3)**2) # m/s
            dopplerW = lambda0/c*dopplerV # Angstrom
            lambdaB = 4.67e-13*lambda0**2*B # Angstrom
            u0 = (lambdas-lambda0)/dopplerW
            uB0 = lambdaB/dopplerW
            Gamma = self.DampingParameter(line_i, T, p)
            a = (Gamma*(lambda0)**2/(4*np.pi*c*1e10*dopplerW))
            kappaL = selective_absorption(T,Pe,dopplerV,line_i,self.AP)
            if (debug := kwargs.get('debug',None)):
                if debug.get('check_kappa'):
                    if 'kappaL' not in self.debug_log.keys():
                        self.debug_log['kappaL']={iline:kappaL}
                    else:
                        self.debug_log['kappaL'][iline]=kappaL
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
                    uB = uB0*(ml*gl-mu*gu)
                    u = (u0+uB-uLos)
                    H,F = voigt2(u,a)
                    if i==0:
                        eta_alpha = Si*H
                        rho_alpha = Si*F*2
                    else:
                        eta_alpha+= Si*H
                        rho_alpha+= Si*F*2
                if alpha==1:
                    ieta_b = eta_alpha*kappaL
                    irho_b = rho_alpha*kappaL
                elif alpha==0:
                    ieta_p = eta_alpha*kappaL
                    irho_p = rho_alpha*kappaL
                else:
                    ieta_r = eta_alpha*kappaL
                    irho_r = rho_alpha*kappaL
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
        if (debug := kwargs.get('debug',None)):
            if debug.get('check_etarho'):
                self.debug_log['etarho']=dict(
                    eta_b = eta_b,
                    eta_p = eta_p,
                    eta_r = eta_r,
                    rho_b = rho_b,
                    rho_p = rho_p,
                    rho_r = rho_r
                )
        return eta_b,eta_p,eta_r,rho_b,rho_p,rho_r

    def DampingParameter(self, line_i, T, p):
        kB = const.k*1e7
        Y_rad = line_i['Y_rad']
        Y_vdw = line_i['Y_vdw']
        M     = line_i['M']
        p_Htot = p["p(H')"]
        p_H    = p["p(H)/p(H')"]*p_Htot
        p_H2   = p["p(H2)/p(H')"]*p_Htot
        p_He   = p["p(He)/p(H')"]*p_Htot
        N_H = p_H/(kB*T)
        N_H2 = p_H2/(kB*T)
        N_He = p_He/(kB*T)
        Ni    = [N_H,N_H2,N_He]
        vdw_contributor = self.van_der_waals_contributor
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

    def _Source_function(self, T, lambdas, **kwargs):
        SI = Planck_Blambda(T, lambdas)
        SQ = torch.zeros_like(SI)
        SU = torch.zeros_like(SI)
        SV = torch.zeros_like(SI)
        S  = torch.stack([SI,SQ,SU,SV], axis=-1)
        if (debug := kwargs.get('debug',None)):
            if debug.get('check_Sfun',None):
                self.debug_log['Sfun']=S
        return S

    def _OpacityInterpolator(self, opacity_type):
        if opacity_type=='formula':
            self.kapp_interp = None
            pass
        elif opacity_type=='Opacity Project':
            # tab_summs = kwargs.get('OPtab_summs', './table_summaries.csv')
            tab_summs = pkg_resources.resource_filename(
                'pyprt',
                'data/OPtab_summaries.csv'
            )
            tab_summs = pd.read_csv(tab_summs)
            XYZ = tab_summs.values[:,1:4]
            X,Y,Z = get_xyz(self.AP)
            OPtab_idx = np.argmin(np.abs(XYZ-np.array([X,Y,Z])[None,:]).sum(axis=1))
            OPtabs_path = pkg_resources.resource_filename('pyprt', 'data/OPtabs')
            OPtabs_file = sorted(glob.glob(os.path.join(OPtabs_path,'*.csv')))[OPtab_idx]
            df_read = pd.read_csv(OPtabs_file)
            logRlist = np.arange(-8.0,1.0+0.5,0.5)
            logTlist = df_read.values[:,0]
            opacitys  = df_read.values[:,1:]
            grid_logR = torch.from_numpy(logRlist)
            grid_logT = torch.from_numpy(logTlist)
            OP_tensor = torch.from_numpy(opacitys)
            OP_interp = TensorGridInterpolator((grid_logT,grid_logR),OP_tensor,bounds_error=False)
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
            raise ValueError(f'`opacity_type` only support "formula","ATLAS", or "Opacity Project", but get {opacity_type}')