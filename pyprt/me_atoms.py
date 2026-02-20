from .needs import *
from .math import *
from .network import *
from .phys import *

class synth_me():
    """
    Synthesis Stokes spectral base on Milne-Eddington atomosphere
    """
    kB = const.k*1e7 # erg/K -> Boltzmann constant
    u  = const.m_p*1e3 # g -> atomic mass unit
    c  = const.c*1e2  # cm/s -> speed of light
    e  = const.e*1e7  # erg -> elementary electron charge
    def __init__(
        self,
        wavebands: torch.Tensor, # [Nw,]
        landeG: float = 2.5, # Landé g factor
        lambda0: float = 6302.5, # spectral reference wavelength in Ångström
        Ar: float = 55.85, # relative atomic mass of corresponding atom
        wing: float = None, # reference line wing in Ångström
        **kwargs
    ):
        '''
        Parameters:
            wavebands: torch.Tensor, # [Nw,]
            landeG: float = 2.5, # Landé g factor
            lambda0: float = 6302.5, # spectral reference wavelength in Ångström
            Ar: float = 55.85, # relative atomic mass of corresponding atom
            wing: float = None, # reference line wing in Ångström
        '''
        self.G = landeG
        self.lambda0 = lambda0
        self.Ar = Ar
        self.wing = wing if wing is not None else wavebands[0]
        self.device = torch.device(kwargs.get('device', wavebands.device))
        self.dtype  = wavebands.dtype
        wavebands = wavebands.to(self.device)
        self.wavebands = torch.cat([
            torch.tensor([self.wing]).to(self.device).type(self.dtype), wavebands
        ])
        self.VoigtNet = torch.load('./VoigtNet/voigtnet_new.pkl', weights_only=False).to(self.device)
        self.VoigtFaradayNet = torch.load('./VoigtFaradayNet/voigtfaraday.pkl', weights_only=False).to(self.device)
        self.checker = dict()

    def __call__(
        self,
        D:torch.Tensor, # [N,]
        V:torch.Tensor, # [N,]
        E:torch.Tensor, # [N,]
        S:torch.Tensor, # [N,]
        A:torch.Tensor, # [N,]
        B:torch.Tensor, # [N,]
        T:torch.Tensor, # [N,]
        P:torch.Tensor, # [N,]
        M:torch.Tensor = None, # [N,] -> Macro turbulence velocity in [km/s]
    ):
        '''
        Parameters:
            D: torch.Tensor, # [N,] -> Doppler width include micro turbulence and thermal motion in [mA]
            V: torch.Tensor, # [N,] -> line of sight velocity [km/s]
            E: torch.Tensor, # [N,] -> eta_0: fraction between selective absorption and continuum absorption []
            S: torch.Tensor, # [N,] -> S1/S0: source function: S = S0+S1*tau []
            A: torch.Tensor, # [N,] -> a parameter for damping factor for absorption profile []
            B: torch.Tensor, # [N,] -> Magnetic field strength B in [G]
            T: torch.Tensor, # [N,] -> Magnetic inclination theta in [deg]
            P: torch.Tensor, # [N,] -> Magnetic azimuth phi in [deg]
            M: torch.Tensor, # [N,] -> Macro turbulence velocity in [km/s], default is None

        Return:
            Stokes = [I,Q,U,V]/Ic # [4,N]
        '''
        D = D.unsqueeze(1).to(device=self.device,dtype=self.dtype)*1e-3 # [A]
        V = V.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        E = E.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        S = S.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        A = A.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        B = B.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        T = T.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        P = P.unsqueeze(1).to(device=self.device,dtype=self.dtype)
        
        u_los = self.lambda0*V*1e5/self.c/D
        lambdaB = 4.67e-13*self.lambda0**2*B
        u_B   = lambdaB/D
        phipsi = self._return_profile(D, u_los, u_B, A)
        etarho = self._propagation_matrix(E,T,P,phipsi)
        etaI,etaQ,etaU,etaV,rhoQ,rhoU,rhoV = etarho
        Delta = etaI**2*(etaI**2-etaQ**2-etaU**2-etaV**2+rhoQ**2+rhoU**2+rhoV**2)-(etaQ*rhoQ+etaU*rhoU+etaV*rhoV)**2
        Delta_r = 1/Delta

        I = 1+Delta_r*etaI*(etaI**2-etaQ**2-etaU**2-etaV**2)*S
        Q = Delta_r*(etaI**2*etaQ+etaI*(etaV*rhoU-etaU*rhoV)+rhoQ*(etaQ*rhoQ+etaU*rhoU+etaV*rhoV))*S
        U = Delta_r*(etaI**2*etaU+etaI*(etaQ*rhoV-etaV*rhoQ)+rhoU*(etaQ*rhoQ+etaU*rhoU+etaV*rhoV))*S
        V = Delta_r*(etaI**2*etaV+etaI*(etaU*rhoQ-etaQ*rhoU)+rhoV*(etaQ*rhoQ+etaU*rhoU+etaV*rhoV))*S

        Ic = I[:,0:1]
        I0 = I[:,1:]/Ic
        Q0 = Q[:,1:]/Ic
        U0 = U[:,1:]/Ic
        V0 = V[:,1:]/Ic
        Stokes = torch.stack([I0,Q0,U0,V0],dim=0) # [4,N,Nw]
        return Stokes
    
    def _Voiget(self,u,a):
        size = u.shape
        self.VoigtNet.eval()
        with torch.no_grad():
            ret = torch.exp(self.VoigtNet(u,a))*np.pi**0.5
        ret = ret.reshape(size)
        return ret
    
    def _VoigtFaraday(self,u,a):
        size = u.shape
        self.VoigtFaradayNet.eval()
        with torch.no_grad():
            ret = self.VoigtFaradayNet(u,a)*np.pi**0.5
        ret = ret.reshape(size)
        return ret

    def _return_profile(self,D,u_los,u_B,A):
        '''
        Calculate the absorption and dispersion profile
        '''
        wavebands = self.wavebands.unsqueeze(0) # [1,Nw]
        u0 = (wavebands-self.lambda0)/D
        size = (D.size(0),wavebands.size(1))
        G  = self.G
        a  = A.repeat(1,size[1]).flatten()
        phi_0 = self._Voiget((u0-u_los      ).flatten(), a).reshape(size)
        phi_b = self._Voiget((u0-u_los+G*u_B).flatten(), a).reshape(size)
        phi_r = self._Voiget((u0-u_los-G*u_B).flatten(), a).reshape(size)
        psi_0 = self._VoigtFaraday((u0-u_los      ).flatten(), a).reshape(size)
        psi_b = self._VoigtFaraday((u0-u_los+G*u_B).flatten(), a).reshape(size)
        psi_r = self._VoigtFaraday((u0-u_los-G*u_B).flatten(), a).reshape(size)
        phipsi = torch.stack([phi_0,phi_b,phi_r,psi_0,psi_b,psi_r], dim=0)
        # self.checker['phipsi'] = phipsi
        return phipsi

    def _propagation_matrix(self,E,T,P,phipsi):
        phi_0,phi_b,phi_r,psi_0,psi_b,psi_r = phipsi
        etaI = 1+E/2*(phi_0*torch.sin(T)**2+0.5*(phi_b+phi_r)*(1+torch.cos(T)**2))
        etaQ = E/2*(phi_0-0.5*(phi_b+phi_r))*torch.sin(T)**2*torch.cos(2*P)
        etaU = E/2*(phi_0-0.5*(phi_b+phi_r))*torch.sin(T)**2*torch.sin(2*P)
        etaV = E/2*(phi_r-phi_b)*torch.cos(T)
        rhoQ = E/2*(psi_0-0.5*(psi_b+psi_r))*torch.sin(T)**2*torch.cos(2*P)
        rhoU = E/2*(psi_0-0.5*(psi_b+psi_r))*torch.sin(T)**2*torch.sin(2*P)
        rhoV = E/2*(psi_r-psi_b)*torch.cos(T)
        etarho = torch.stack([etaI,etaQ,etaU,etaV,rhoQ,rhoU,rhoV],dim=0)
        # self.checker['etarho'] = etarho
        return etarho
