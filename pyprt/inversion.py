from pyprt.needs import *
from pyprt import synth
from pyprt.utils import load_initial_guess,grid2node,node2grid,spline_smooth
import time
import pickle
from CuSP.annealing import DualAnnealing

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
        self.log = dict()

    def __call__(
        self,
        iquv_obs: torch.Tensor, # (Nb,Nw,4)
        wavs: torch.Tensor, # (Nw,)
        initial_guess="Quiet_sun",
        device=None,
        mode='fix_pe', # fix_pe,fix_pg,hse
        maxiters=20,
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
        device = iquv_obs.device if device is None else device
        wavs   = wavs.float().to(device=device)
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
        self.device=device
        pe_fixed = kwargs.pop('pe_fixed',None)
        Pe0 = Pe0 if pe_fixed is None else pe_fixed
        if mode=="fix_pe":
            ivs = self._main(iquv_obs,wavs,ltau,param0,pe_fixed=Pe0,maxiters=maxiters,**kwargs)
        elif mode=='fix_pg':
            ivs = self._main(iquv_obs,wavs,ltau,param0,pg_fixed=Pg0,maxiters=maxiters,**kwargs)
        elif mode=='hse':
            ivs = self._main(iquv_obs,wavs,ltau,param0,maxiters=maxiters,**kwargs)
        else:
            raise ValueError(f"mode {mode} not implemented")
        return ivs

    def _main(self,iquv_obs,wavs,ltau,param0,pe_fixed=None,pg_fixed=None,maxiters=20,**kwargs):
        Nt = ltau.size(0)
        Nb = iquv_obs.size(0)
        Nw = iquv_obs.size(1)
        nnodes_list = kwargs.get('nnodes_list',self._nnodes(Nt))
        self.Nb,self.Nt,self.Nw = Nb,Nt,Nw
        self.DOF = Nw*4-self.Np # Degree Of Freedom
        smooth = kwargs.get('smooth',True)

        device = self.device
        x0 = param0.clone().unsqueeze(0).repeat(Nb,1)
        time0 = time.time()
        for i,nnodes in enumerate(nnodes_list):
            if i==0:
                xnodes = grid2node(ltau,x0,nnodes)
            else:
                xnodes = grid2node(ltau,xgrid,nnodes)
            def func(xnodes):
                xgrid = node2grid(ltau,xnodes,nnodes)
                pe = self._get_pe(ltau,xgrid,pe_fixed=pe_fixed,pg_fixed=pg_fixed)
                iquv_syn = self._syn(wavs,ltau,xgrid,pe=pe)
                chi2 = self._Chi2(iquv_syn,iquv_obs)
                return chi2
            bounds = [[2e3,1e4]]*nnodes+[[-5,5]]*nnodes+[[0,4e3]]*nnodes+[[0,np.pi]]*2*nnodes+[[0.0,2]]*2
            # maxiter = max(maxiters,100) if i<=2 else maxiters
            maxiter = maxiters
            annealing_config = kwargs.get('annealing',dict())
            adam = annealing_config.pop('adam',dict(ls_max_iter=20))
            _ = annealing_config.pop('maxiter',None)
            _ = annealing_config.pop('bounds',None)
            save_path = annealing_config.pop('save_path','./Annealing/')
            os.makedirs(save_path,exist_ok=True)
            save_name = annealing_config.pop('save_name','result')
            save_file = os.path.join(save_path,f'{save_name}.{i+1:04d}.N{nnodes:04d}.pkl')
            if i==0:
                hist_files = [save_file]
            else:
                hist_files.append(save_file)
            res = DualAnnealing(
                func,bounds,maxiter=maxiter,x0=xnodes,
                adam=adam,
                **annealing_config
                )
            with open(save_file,'wb') as f:
                pickle.dump(res,f)
            chi2 = res.e
            message = f"Step: {i+1:4d}/{len(nnodes_list)} | "
            message+= f"Nodes: {nnodes:4d}/{np.max(nnodes_list)} | Chi2: {chi2.item():.4e} | "
            message+= f"Time : {(time.time()-time0)/60:6.2f} min "
            print(message)
            xgrid = node2grid(ltau,res.x,nnodes)
            if nnodes>4 and smooth:
                xgrid = self._smooth(ltau,xgrid)
        t = xgrid[:,0*Nt:1*Nt].detach().cpu().numpy()
        v = xgrid[:,1*Nt:2*Nt].detach().cpu().numpy()
        b = xgrid[:,2*Nt:3*Nt].detach().cpu().numpy()
        g = xgrid[:,3*Nt:4*Nt].detach().cpu().numpy()
        f = xgrid[:,4*Nt:5*Nt].detach().cpu().numpy()
        m = xgrid[:,5*Nt     ].detach().cpu().numpy()
        M = xgrid[:,5*Nt+1   ].detach().cpu().numpy()
        p = self._get_pe(ltau,xgrid,pe_fixed=pe_fixed,pg_fixed=pg_fixed).detach().cpu().numpy()
        ivs = dict(
            T=t,vLos=v,Pe=p,Bmag=b,inclination=g,azimuth=f,vmic=m,vmac=M,
            ltau=ltau.detach().cpu().numpy().ravel(),
            res=res,
        )
        try:
            self.log['annealing']=dict(annealing_hist=hist_files,nnodes_list=nnodes_list)
        except:
            self.log=dict(
                annealing=dict(
                    annealing_hist=hist_files,nnodes_list=nnodes_list
                )
                )
        return ivs

    @staticmethod
    def _smooth(ltau,xgrid):
        device = xgrid.device
        Nb = xgrid.size(0)
        Nt = ltau.numel()
        ltau_array = ltau.detach().squeeze().cpu().numpy()
        # print('1: ',xgrid.shape)
        for i in range(Nb):
            ti = xgrid[i,0*Nt:1*Nt].detach().cpu().numpy()
            vi = xgrid[i,1*Nt:2*Nt].detach().cpu().numpy()
            bi = xgrid[i,2*Nt:3*Nt].detach().cpu().numpy()
            gi = xgrid[i,3*Nt:4*Nt].detach().cpu().numpy()
            fi = xgrid[i,4*Nt:5*Nt].detach().cpu().numpy()
            mi = xgrid[i,5*Nt     ].detach().cpu().numpy()
            Mi = xgrid[i,5*Nt+1   ].detach().cpu().numpy()
            ti,_ = spline_smooth(ltau_array,ti,adaptive=True)
            vi,_ = spline_smooth(ltau_array,vi,adaptive=True)
            bi,_ = spline_smooth(ltau_array,bi,adaptive=True)
            gi,_ = spline_smooth(ltau_array,gi,adaptive=True)
            fi,_ = spline_smooth(ltau_array,fi,adaptive=True)
            # print('2: ',ti.shape)
            xsmooth = torch.from_numpy(np.concatenate([ti,vi,bi,gi,fi,[mi],[Mi]],axis=0)).float()
            # print('3: ',xsmooth.shape,xgrid[i].shape)
            xgrid[i] = xsmooth.clone().to(device)
            del xsmooth,ti,vi,bi,gi,fi,mi,Mi
        return xgrid

    @staticmethod
    def _nnodes(Nt):
        nums = np.floor(np.log(Nt)/np.log(2))
        nnodes_list = [int(2**i) for i in range(int(nums))]
        nnodes_list.append(Nt)
        return nnodes_list

    def _main_old(self,iquv_obs,wavs,ltau,param0,pe_fixed=None,pg_fixed=None,maxiters=1000,**kwargs):
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
        for i in range(maxiters):
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
            # print(self.Nb)
            pe = pe_fixed.unsqueeze(0).repeat(self.Nb,1) # (Nb,Nt)
            # print(pe_fixed.shape)
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
        Ic = iquv_obs[:,0:1,0:1] # (Nb,1,1)
        F  = self.DOF
        Ws = Ws[None,:] # (1,4)
        Ss = Ss[None,None,:] # (1,1,4)
        chi2 = torch.sum(
            Ws**2*torch.sum((iquv_syn - iquv_obs)**2/Ss**2/Ic**2,dim=1),
            dim=-1,
            keepdim=True
            )/F # (Nb,)
        return chi2
