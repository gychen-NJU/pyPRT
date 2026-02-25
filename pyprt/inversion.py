from pyprt.needs import *
from pyprt import synth
from pyprt.utils import load_initial_guess

class Inversion(synth):
    """
    Stokes inversion
    """
    def __init__(
        self,
        table_abu = None,
        solver = 'Hermitian',
        refidx = 1., # refractive index
        lines = {"FeI":["6301.5","6302.5"]},
        opacity_type='formula', # 'formula', 'ATLAS', 'Opacity Project'
        w_iquv = [1.,5.,5.,3.5]
    ):
        super().__init__(
            table_abu = table_abu,
            solver = solver,
            refidx = refidx,
            lines = lines,
            opacity_type = opacity_type
        )
        self.w_iquv = w_iquv
        self.DOF = None
        self.Nt = None
        self.Nb = None
        self.Nw = None

    def __call__(
        self,
        iquv_obs: torch.Tensor, # (Nb,Nw,4)
        wavs: torch.Tensor, # (Nw,)
        initial_guess="Quiet_sun",
        device=None,
        mode='fix_pe', # fix_pe,fix_pg,hse
        max_steps=1000,
        **kwargs
    ):
        """
        Inversion
        
        Parameters:
        -------------
            initial_guess: str, optional
                Initial guess for the inversion. Default is "Quiet_sun".
        """
        IG = load_initial_guess(initial_guess)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        ltau   = torch.from_numpy(IG["ltau"]       ).flip(0).float().to(device=device)
        T0     = torch.from_numpy(IG["T"]          ).flip(0).float().to(device=device)
        Pe0    = torch.from_numpy(IG["Pe"]         ).flip(0).float().to(device=device)
        vmic0  = torch.from_numpy(IG["vmic"]       ).flip(0).float().to(device=device)
        B0     = torch.from_numpy(IG["Bmag"]       ).flip(0).float().to(device=device)
        vLos0  = torch.from_numpy(IG["vLos"]       ).flip(0).float().to(device=device)
        gamma0 = torch.from_numpy(IG["inclination"]).flip(0).float().to(device=device)
        phi0   = torch.from_numpy(IG["azimuth"]    ).flip(0).float().to(device=device)
        Pg0    = torch.from_numpy(IG["Pg"]         ).flip(0).float().to(device=device)
        vmac0  = torch.tensor([0.]).float().to(device=device)
        param0 = torch.cat([T0,vLos0,B0,gamma0,phi0,vmic0[0:1],vmac0],dim=0)
        self.Np = param0.numel() # Number of parameters
        if mode=="fix_pe":
            ivs = self._main(iquv_obs,wavs,ltau,param0,pe_fixed=Pe0,max_steps=max_steps,**kwargs)
        elif mode=='fix_pg':
            ivs = self._main(iquv_obs,wavs,ltau,param0,pg_fixed=Pg0,max_steps=max_steps,**kwargs)
        elif mode=='hse':
            ivs = self._main(iquv_obs,wavs,ltau,param0,max_steps=max_steps,**kwargs)
        else:
            raise ValueError(f"mode {mode} not implemented")
        return ivs

    def _main(self,iquv_obs,wavs,ltau,param0,pe_fixed=None,pg_fixed=None,max_steps=1000,**kwargs):
        """
        Main function for inversion
        
        Parameters:
        -------------
            iquv_obs: torch.Tensor # (Nb,Nw,4)
            wavs: torch.Tensor # (Nw,)
            ltau: torch.Tensor # (Nt,)
            param0: torch.Tensor # (DOF,)
            pe_fixed: torch.Tensor # (Nt,)
            pg_fixed: torch.Tensor # (Nt,)
        """
        Adam_config = kwargs.get('Adam_config',dict(lr=1.e-3))
        Nt = ltau.size(0)
        Nb = iquv_obs.size(0)
        Nw = iquv_obs.size(1)
        param = param0.clone().unsqueeze(0).repeat(Nb,1)
        optimizer = torch.optim.Adam(param, **Adam_config)
        chi2_hist = []
        self.Nb,self.Nt,self.Nw = Nb,Nt,Nw
        self.DOF = Nw*4-self.Np # Degree Of Freedom
        for i in range(max_steps):
            param.requires_grad_(True)
            pe = self._get_pe(ltau,param,pe_fixed=pe_fixed,pg_fixed=pg_fixed)
            iquv_syn = self._syn(wavs,ltau,param,pe=pe)
            chi2 = self._Chi2(iquv_syn,iquv_obs)
            chi2_hist.append(chi2.detach().cpu().numpy())
            optimizer.zero_grad()
            chi2.backward()
            optimizer.step()
        chi2_hist = np.stack(chi2_hist,axis=1) # (Nb,max_steps)
        self.chi2_hist = chi2_hist
        t = param[:,0*Nt:1*Nt].detach().cpu().numpy()
        v = param[:,1*Nt:2*Nt].detach().cpu().numpy()
        b = param[:,2*Nt:3*Nt].detach().cpu().numpy()
        g = param[:,3*Nt:4*Nt].detach().cpu().numpy()
        f = param[:,4*Nt:5*Nt].detach().cpu().numpy()
        m = param[:,5*Nt     ].detach().cpu().numpy()
        M = param[:,5*Nt+1   ].detach().cpu().numpy()
        p = self._get_pe(ltau,param,pe_fixed=pe_fixed,pg_fixed=pg_fixed).detach().cpu().numpy()
        ivs = dict(
            T=t,vLos=v,Pe=p,Bmag=b,inclination=g,azimuth=f,vmic=m,vmac=M,
        )
        return ivs

    def _get_pe(self,ltau,param,pe_fixed=None,pg_fixed=None):
        """
        Get Pe from param
        
        Parameters:
        -------------
            ltau: torch.Tensor # (Nt,)
            param: torch.Tensor # (Nb,DOF)
            pe_fixed: None or torch.Tensor # (Nt,)
            pg_fixed: None or torch.Tensor # (Nt,)
        """
        if pe_fixed is not None:
            pe = pe_fixed.unsqueeze(0).repeat(Nb,1) # (Nb,Nt)
        if pg_fixed is not None:
            pass
        else:
            pass
        return pe

    def _syn(self,wavs,ltau,param,pe=None):
        """
        Get synthetic IQUV from param
        
        Parameters:
        -------------
            wavs: torch.Tensor # (Nw,)
            ltau: torch.Tensor # (Nt,)
            param: torch.Tensor # (Nb,DOF)
            pe: None or torch.Tensor # (Nt,)
        """
        Nb,Nt = self.Nb,self.Nt
        p = pe # (Nb,Nt)
        t = param[:,0*Nt:1*Nt] #(Nb,Nt)
        v = param[:,1*Nt:2*Nt] #(Nb,Nt)
        b = param[:,2*Nt:3*Nt] #(Nb,Nt)
        g = param[:,3*Nt:4*Nt] #(Nb,Nt)
        f = param[:,4*Nt:5*Nt] #(Nb,Nt)
        m = param[:,5*Nt     ][:,None].repeat(1,Nt) #(Nb,Nt)
        M = param[:,5*Nt+1   ] #(Nb,)
        iquv_syn = super().__call__(wavs,ltau,t,p,b,g,f,v,m,M)
        return iquv_syn

    def _Chi2(self, iquv_syn, iquv_obs):
        """
        Calculate Chi2 between synthetic and observed IQUV
        
        Parameters:
        -------------
            iquv_syn: torch.Tensor # (Nb,Nw,4)
            iquv_obs: torch.Tensor # (Nb,Nw,4)
        """
        Ws = torch.tensor(self.w_iquv, dtype=iquv_syn.dtype, device=iquv_syn.device)
        Ss = torch.tensor([0.118]+[0.204]*3,dtype=iquv_obs.dtype,device=iquv_obs.device)
        F  = self.DOF
        Ws = Ws[None,:] # (1,4)
        Ss = Ss[None,None,:] # (1,1,4)
        chi2 = torch.sum(
            Ws**2*torch.sum((iquv_syn - iquv_obs)**2/Ss**2,dim=1),
            dim=-1
            )/F # (Nb,)
        return chi2
