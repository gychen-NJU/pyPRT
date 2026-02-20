from .needs import *

handlers = {}

def register_handler(key):
    def decorator(func):
        handlers[key] = func
        return func
    return decorator

@register_handler(1)
def u123_H(T):
    u1 = torch.full_like(T,2.)
    u1 = torch.where(T>1.3e4,1.51+3.8e-5*T,u1)
    u1 = torch.where(T>1.62e4,11.41+T*(-1.1428e-3+T*3.52e-8),u1)
    u2 = torch.ones_like(T)
    u3 = torch.zeros_like(T)
    return u1,u2,u3

@register_handler(2)
def u123_He(T):
    u1 = torch.ones_like(T)
    u1 = torch.where(T>3e4,14.8+T*(-9.4103e-4+T*1.6095e-8),u1)
    u2 = torch.full_like(T,2.0)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(3)
def u123_Li(T):
    # X = torch.log(5040/T)
    T3 = 1.e-3*T
    u1 = 2.081-T3*(6.8926E-2-T3*1.4081E-2)
    u1 = torch.where(T>6e3,3.4864+T*(-7.3292E-4+T*8.5586E-8),u1)
    u2 = torch.ones_like(T)
    u3 = torch.full_like(T,2.)
    return u1,u2,u3

@register_handler(4)
def u123_Be(T):
    u1 = torch.clamp(.631+7.032E-5*T,min=1.)
    u2 = torch.full_like(T,2.0)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(5)
def u123_B(T):
    T3 = 1.e-3*T
    u1 = 5.9351+1.0438E-2*T3
    u2 = torch.ones_like(T)
    u3 = torch.full_like(T,2.)
    return u1,u2,u3

@register_handler(6)
def u123_C(T):
    T3 = 1.e-3*T
    u1 = 8.6985+T3*(2.0485E-2+T3*(1.7629E-2-3.9091E-4*T3))
    u1 = torch.where(T>1.2e4,12.54+T*(-1.2933e-3+T*4.20e-8),u1)
    u2 = 5.838+1.6833E-5*T
    u2 = torch.where(T>2.4e4,10.989+T*(-6.9347E-4+T*2.0861E-8),u2)
    u3 = torch.ones_like(T)
    u3 = torch.where(T>1.95e4,-0.555+8E-5*T,u3)
    return u1,u2,u3

@register_handler(7)
def u123_N(T):
    T3 = 1.e-3*T
    u1 = 3.9914+T3*(1.7491E-2-T3*(1.0148E-2-T3*1.7138E-3))
    u1 = torch.where(T>8800,2.171+2.54E-4*T,u1)
    u1 = torch.where(T>1.8e4,11.396+T*(-1.7139E-3+T*8.633E-8),u1)
    u2 = 8.060+1.420E-4*T
    u2 = torch.where(T>3.3e4,26.793+T*(-1.8931E-3+T*4.4612E-8),u2)
    u3 = 5.9835+T*(-2.6651E-5+T*1.8228E-9)
    u3 = torch.where(T<7310.5,5.89,u3)
    return u1,u2,u3

@register_handler(8)
def u123_O(T):
    u1 = 8.29+1.10E-4*T
    u1 = torch.where(T>1.9e4,66.81+T*(-6.019E-3+T*1.657E-7),u1)
    u2 = torch.clamp(3.51+8.E-5*T,min=4.0)
    u2 = torch.where(T>3.64e4,68.7+T*(-4.216E-3+T*6.885E-8),u2)
    u3 = 7.865+1.1348E-4*T
    return u1,u2,u3

@register_handler(9)
def u123_F(T):
    T3 = 1.e-3*T
    u1 = 4.5832+T3*(.77683+T3*(-.20884+T3*(2.6771E-2-1.3035E-3*T3)))
    u1 = torch.where(T>8750,5.9,u1)
    u1 = torch.where(T>2.e4,15.16+T*(-9.229E-4+T*2.312E-8),u1)
    u2 = 8.15+8.9E-5*T
    u3 = 2.315+1.38E-4*T
    return u1,u2,u3

@register_handler(10)
def u123_Ne(T):
    u1 = torch.ones_like(T)
    u1 = torch.where(T<2.69e4,26.3+T*(-2.113E-3+T*4.359E-8),u1)
    u2 = 5.4+4.E-5*T
    u3 = 7.973+7.956E-5*T
    return u1,u2,u3

@register_handler(11)
def u123_Na(T):
    u1 = torch.clamp(1.72+9.3E-5*T,min=2.)
    u1 = torch.where(T>5400,-0.83+5.66E-4*T,u1)
    u1 = torch.where(T>8.5e3,4.5568+T*(-1.2415E-3+T*1.3861E-7),u1)
    u2 = torch.ones_like(T)
    u3 = 5.69+5.69E-6*T
    return u1,u2,u3

@register_handler(12)
def u123_Mg(T):
    X = torch.log(5040/T)
    u1 = 1.+torch.exp(-4.027262-X*(6.173172+X*(2.889176+X*(2.393895+.784131*X))))
    u1 = torch.where(T>8.e3,2.757+T*(-7.8909E-4+T*7.4531E-8),u1)
    u2 = 2.+torch.exp(-7.721172-X*(7.600678+X*(1.966097+.212417*X)))
    u2 = torch.where(T>2e4,7.1041+T*(-1.0817E-3+T*4.7841E-8),u2)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(13)
def u123_Al(T):
    T3 = 1.e-3*T
    u1 = 5.2955+T3*(.27833-T3*(4.7529E-2-T3*3.0199E-3))
    u2 = torch.clamp(.725+3.245E-5*T,min=1)
    u2 = torch.where(T>2.24e4,61.06+T*(-5.987E-3+T*1.485E-7),u2)
    u3 = torch.clamp(1.976+3.43E-6*T,2)
    u3 = torch.where(T>1.814e4,3.522+T*(-1.59E-4+T*4.382E-9),u3)
    return u1,u2,u3

@register_handler(14)
def u123_Si(T):
    T3 = 1.e-3*T
    u1 = 6.7868+T3*(.86319+T3*(-.11622+T3*(.013109-6.2013E-4*T3)))
    u1 = torch.where(T>1.04e4,86.01+T*(-1.465E-2+T*7.282E-7),u1)
    u2 = 5.470+4.E-5*T
    u2 = torch.where(T>1.8e4,26.44+T*(-2.22E-3+T*6.188E-8),u2)
    u3 = torch.clamp(.911+1.1E-5*T, 1.)
    u3 = torch.where(T>3.3e4,19.14+T*(-1.408E-3+T*2.617E-8),u3)
    return u1,u2,u3

@register_handler(15)
def u123_P(T):
    T3 = 1.e-3*T
    u1 = 4.2251+T3*(-.22476+T3*(.057306-T3*1.0381E-3))
    u1 = torch.where(T>6e3,1.56+5.2E-4*T,u1)
    u2 = 4.4151+T3*(2.2494+T3*(-.55371+T3*(.071913-T3*3.5156E-3)))
    u2 = torch.where(T>7250,4.62+5.38E-4*T,u2)
    u3 = 5.595+3.4E-5*T
    return u1,u2,u3

@register_handler(16)
def u123_S(T):
    u1 = 7.5+2.15E-4*T
    u1 = torch.where(T>1.16e4,38.76+T*(-4.906E-3+T*2.125E-7),u1)
    u2 = 2.845+2.43E-4*T
    u2 = torch.where(T>1.05e4,6.406+T*(-1.68E-4+T*1.323E-8),u2)
    u3 = 7.38+1.88E-4*T
    return u1,u2,u3

@register_handler(17)
def u123_Cl(T):
    u1 = 5.2+6.E-5*T
    u1 = torch.where(T>1.84e4,-81.6+4.8E-3*T,u1)
    u2 = 7.0+2.43E-4*T
    u3 = 2.2+2.62E-4*T
    return u1,u2,u3

@register_handler(18)
def u123_Ar(T):
    u1 = torch.ones_like(T)
    u2 = 5.20+3.8E-5*T
    u3 = 7.474+1.554E-4*T
    return u1,u2,u3

@register_handler(19)
def u123_K(T):
    T3 = 1.e-3*T
    u1 = 1.9909+T3*(.023169-T3*(.017432-T3*4.0938E-3))
    u1 = torch.where(T>5800,-9.93+2.124E-3*T,u1)
    u2 = torch.ones_like(T)
    u3 = 5.304+1.93E-5*T
    return u1,u2,u3

@register_handler(20)
def u123_Ca(T):
    X = torch.log(5040/T)
    u1 = 1.+torch.exp(-1.731273-X*(5.004556+X*(1.645456+X*(1.326861+.508553*X))))
    u2 = 2.+torch.exp(-1.582112-X*(3.996089+X*(1.890737+.539672*X)))
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(21)
def u123_Sc(T):
    X = torch.log(5040/T)
    u1 = 4.+torch.exp(2.071563+X*(-1.2392+X*(1.173504+.517796*X)))
    u2 = 3.+torch.exp(2.988362+X*(-.596238+.054658*X))
    u3 = torch.full_like(T,10)
    return u1,u2,u3

@register_handler(22)
def u123_Ti(T):
    X = torch.log(5040/T)
    u1 = 5.+torch.exp(3.200453+X*(-1.227798+X*(.799613+.278963*X)))
    u1 = torch.where(T<5.5e3,16.37+T*(-2.838E-4+T*5.819E-7),u1)
    u2 = 4.+torch.exp(3.94529+X*(-.551431+.115693*X))
    u3 = 16.4+8.5E-4*T
    return u1,u2,u3

@register_handler(23)
def u123_V(T):
    X = torch.log(5040/T)
    u1 = 4.+torch.exp(3.769611+X*(-.906352+X*(.724694+.1622*X)))
    u2 = 1.+torch.exp(3.755917+X*(-.757371+.21043*X))
    u3 = -18.+1.03E-2*T
    u3 = torch.where(T<2.25e3,2.4E-3*T,u3)
    return u1,u2,u3

@register_handler(24)
def u123_Cr(T):
    X = torch.log(5040/T)
    u1 = 7.+torch.exp(1.225042+X*(-2.923459+X*(.154709+.09527*X)))
    u2 = 6.+torch.exp(.128752-X*(4.143973+X*(1.096548+.230073*X)))
    u3 = 10.4+2.1E-3*T
    return u1,u2,u3

@register_handler(25)
def u123_Mn(T):
    X = torch.log(5040/T)
    u1 = 6.+torch.exp(-.86963-X*(5.531252+X*(2.13632+X*(1.061055+.265557*X))))
    u2 = 7.+torch.exp(-.282961-X*(3.77279+X*(.814675+.159822*X)))
    u3 = torch.full_like(T,10)
    return u1,u2,u3

@register_handler(26)
def u123_Fe(T):
    X = torch.log(5040/T)
    u1 = 9.+torch.exp(2.930047+X*(-.979745+X*(.76027+.118218*X)))
    u1 = torch.where(T<4e3,15.85+T*(1.306E-3+T*2.04E-7),u1)
    u1 = torch.where(T>9e3,39.149+T*(-9.5922E-3+T*1.2477E-6),u1)
    u2 = 10.+torch.exp(3.501597+X*(-.612094+.280982*X))
    u2 = torch.where(T>1.8e4,68.356+T*(-6.1104E-3+T*5.1567E-7),u2)
    u3 = 17.336+T*(5.5048E-4+T*5.7514E-8)
    return u1,u2,u3

@register_handler(27)
def u123_Co(T):
    u1 = 8.65+4.9E-3*T
    u2 = 11.2+3.58E-3*T
    u3 = 15.0+1.42E-3*T
    return u1,u2,u3

@register_handler(28)
def u123_Ni(T):
    X = torch.log(5040/T)
    u1 = 9.+torch.exp(3.084552+X*(-.401323+X*(.077498-.278468*X)))
    u2 = 6.+torch.exp(1.593047-X*(1.528966+.115654*X))
    u3 = 13.3+6.9E-4*T
    return u1,u2,u3

@register_handler(29)
def u123_Cu(T):
    u1 = torch.clamp(1.50+1.51E-4*T,min=2)
    u1 = torch.where(T>6250,-.3+4.58E-4*T,u1)
    u2 = torch.clamp(-.3+4.58E-4*T,min=1)
    u3 = 8.025+9.4E-5*T
    return u1,u2,u3

@register_handler(30)
def u123_Zn(T):
    u1 = torch.clamp(.632+5.11E-5*T,min=1)
    u2 = torch.full_like(T,2.)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(31)
def u123_Ga(T):
    T3 = 1.e-3*T
    u1 = 1.7931+T3*(1.9338+T3*(-.4643+T3*(.054876-T3*2.5054E-3)))
    u1 = torch.where(T>6e3,4.18+2.03E-4*T,u1)
    u2 = torch.ones_like(T)
    u3 = torch.full_like(T,2)
    return u1,u2,u3

@register_handler(32)
def u123_Ge(T):
    u1 = 6.12+4.08E-4*T
    u2 = 3.445+1.78E-4*T
    u3 = torch.full_like(T,1.1)
    return u1,u2,u3

@register_handler(33)
def u123_As(T):
    T3 = 1.e-3*T
    u1 = 2.65+3.65E-4*T
    u2 = -.25384+T3*(2.284+T3*(-.33383+T3*(.030408-T3*1.1609E-3)))
    u2 = torch.where(T>1.2e4,8,u1)
    u3 = torch.full_like(T,8)
    return u1,u2,u3

@register_handler(34)
def u123_Se(T):
    T3 = 1.e-3*T
    u1 = 6.34+1.71E-4*T 
    u2 = 4.1786+T3*(-.15392+T3*3.2053E-2)
    u3 = torch.full_like(T,8)
    return u1,u2,u3

@register_handler(35)
def u123_Br(T):
    u1 = 4.12+1.12E-4*T
    u2 = 5.22+3.08E-4*T
    u3 = 2.3+2.86E-4*T
    return u1,u2,u3

@register_handler(36)
def u123_Kr(T):
    u1 = torch.ones_like(T)
    u2 = 4.11+7.4E-5*T
    u3 = 5.35+2.23E-4*T
    return u1,u2,u3

@register_handler(37)
def u123_Rb(T):
    u1 = torch.clamp(1.38+1.94E-4*T,min=2)
    u1 = torch.where(T>6250,-14.9+2.79E-3*T,u1)
    u2 = torch.ones_like(T)
    u3 = 4.207+4.85E-5*T
    return u1,u2,u3

@register_handler(38)
def u123_Sr(T):
    T3 = 1.e-3*T
    u1 = .87127+T3*(.20148+T3*(-.10746+T3*(.021424-T3*1.0231E-3)))
    u1 = torch.where(T>6500,-6.12+1.224E-3*T,u1)
    u2 = torch.clamp(.84+2.6e-4*T,min=2)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(39)
def u123_Y(T):
    u1 = .2+2.58E-3*T
    u2 = 7.15+1.855E-3*T
    u3 = 9.71+9.9E-5*T
    return u1,u2,u3

@register_handler(40)
def u123_Zr(T):
    X = torch.log(5040/T)
    u1 = 76.31+T*(-1.866E-2+T*2.199E-6)
    u1 = torch.where(T>6236,6.8+T*(2.806E-3+T*5.386E-7),u1)
    u2 = 4.+torch.exp(3.721329-.906502*X)
    u3 = 12.3+1.385E-3*T
    return u1,u2,u3

@register_handler(41)
def u123_Nb(T):
    u1 = torch.clamp(-19.+1.43E-2*T,min=1)
    u2 = -4.+1.015E-2*T
    u3 = torch.full_like(T,25)
    return u1,u2,u3

@register_handler(42)
def u123_Mo(T):
    u1 = torch.clamp(2.1+1.5E-3*T,7)
    u1 = torch.where(T>7.e3,-38.1+7.28E-3*T,u1)
    u2 = 1.25+1.17E-3*T
    u2 = torch.where(T>6900,-28.5+5.48E-3*T,u2)
    u3 = 24.04+1.464E-4*T
    return u1,u2,u3

@register_handler(43)
def u123_Tc(T):
    T3 = 1.e-3*T
    u1 = 4.439+T3*(.30648+T3*(1.6525+T3*(-.4078+T3*(.048401-T3*2.1538E-3))))
    u1 = torch.where(T>6e3,24,u1)
    u2 = 8.1096+T3*(-2.963+T3*(2.369+T3*(-.502+T3*(.049656-T3*1.9087E-3))))
    u2 = torch.where(T>6e3,17,u2)
    u3 = torch.full_like(T,220)
    return u1,u2,u3

@register_handler(44)
def u123_Ru(T):
    u1 = -3.+7.17E-3*T
    u2 = 3.+4.26E-3*T
    u3 = torch.full_like(T,22)
    return u1,u2,u3

@register_handler(45)
def u123_Rh(T):
    T3 = 1.e-3*T
    u1 = 6.9164+T3*(3.8468+T3*(.043125-T3*(8.7907E-3-T3*5.9589E-4)))
    u2 = 7.2902+T3*(1.7476+T3*(-.038257+T3*(2.014E-3+T3*2.1218E-4)))
    u3 = torch.full_like(T,30)
    return u1,u2,u3

@register_handler(46)
def u123_Pd(T):
    u1 = torch.clamp(12.6+1.26E-3*T,min=1)
    u2 = 5.60+3.62E-4*T
    u3 = torch.full_like(T,20)
    return u1,u2,u3

@register_handler(47)
def u123_Ag(T):
    u1 = torch.clamp(1.537+7.88E-5*T,min=2)
    u2 = torch.clamp(0.73+3.4E-5*T,min=1)
    u3 = 6.773+1.248E-4*T
    return u1,u2,u3

@register_handler(48)
def u123_Cd(T):
    u1 = torch.clamp(.43+7.6E-5*T,min=1)
    u2 = torch.full_like(T,2)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(49)
def u123_In(T):
    u1 = 2.16+3.92E-4*T
    u2 = torch.ones_like(T)
    u3 = torch.full_like(T,2)
    return u1,u2,u3

@register_handler(50)
def u123_Sn(T):
    u1 = 2.14+6.16E-4*T 
    u2 = 2.06+2.27E-4*T
    u3 = torch.full_like(T,1.05)
    return u1,u2,u3

@register_handler(51)
def u123_Sb(T):
    u1 = 2.34+4.86E-4*T
    u2 = .69+5.36E-4*T
    u3 = torch.full_like(T,3.5)
    return u1,u2,u3

@register_handler(52)
def u123_Te(T):
    T3 = 1.e-3*T
    u1 = 3.948+4.56E-4*T
    u2 = 4.2555+T3*(-.25894+T3*(.06939-T3*2.4271E-3))
    u2 = torch.where(T>1.2e4,7,u2)
    u3 = torch.full_like(T,5)
    return u1,u2,u3

@register_handler(53)
def u123_I(T):
    u1 = torch.clamp(3.8+9.5E-5*T,min=4)
    u2 = 4.12+3.E-4*T
    u3 = torch.full_like(T,7)
    return u1,u2,u3

@register_handler(54)
def u123_Xe(T):
    u1 = torch.ones_like(T)
    u2 = 3.75+6.876E-5*T
    u3 = 4.121+2.323E-4*T
    return u1,u2,u3

@register_handler(55)
def u123_Cs(T):
    u1 = torch.clamp(1.56+1.67E-4*T,min=2)
    u1 = torch.where(T>4850,-2.680+1.04E-3*T,u1)
    u2 = torch.ones_like(T)
    u3 = 3.769+4.971E-5*T
    return u1,u2,u3

@register_handler(56)
def u123_Ba(T):
    u1 = torch.clamp(-1.8+9.85E-4*T,min=1)
    u1 = torch.where(T>6850,-16.2+3.08E-3*T,u1)
    u2 = 1.11+5.94E-4*T
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(57)
def u123_La(T):
    u1 = 15.42+9.5E-4*T
    u1 = torch.where(T>5060,1.+3.8E-3*T,u1)
    u2 = 13.2+3.56E-3*T
    u3 = torch.full_like(T,12)
    return u1,u2,u3

@register_handler(58)
def u123_Ce(T):
    X = torch.log(5040/T)
    u1 = 9.+torch.exp(5.202903+X*(-1.98399+X*(.119673+.179675*X)))
    u2 = 8.+torch.exp(5.634882-X*(1.459196+X*(.310515+.052221*X)))
    u3 = 9.+torch.exp(3.629123-X*(1.340945+X*(.372409+X*(.03186-.014676*X))))
    return u1,u2,u3

@register_handler(59)
def u123_Pr(T):
    X = torch.log(5040/T)
    u2 = 9.+torch.exp(4.32396-X*(1.191467+X*(.149498+.028999*X)))
    u1 = u2
    u3 = 10.+torch.exp(3.206855+X*(-1.614554+X*(.489574+.277916*X)))
    return u1,u2,u3

@register_handler(60)
def u123_Nd(T):
    X = torch.log(5040/T)
    u1 = 9.+torch.exp(4.456882+X*(-2.779176+X*(.082258+X*(.50666+.127326*X))))
    u2 = 8.+torch.exp(4.689643+X*(-2.039946+X*(.17193+X*(.26392+.038225*X))))
    u3 = u2
    return u1,u2,u3

@register_handler(61)
def u123_Pm(T):
    u1 = torch.full_like(T,20)
    u2 = torch.full_like(T,25)
    u3 = torch.full_like(T,100)
    return u1,u2,u3

@register_handler(62)
def u123_Sm(T):
    X = torch.log(5040/T)
    u1 = 1.+torch.exp(3.549595+X*(-1.851549+X*(.9964+.566263*X)))
    u2 = 2.+torch.exp(4.052404+X*(-1.418222+X*(.358695+.161944*X)))
    u3 = 1.+torch.exp(3.222807-X*(.699473+X*(-.056205+X*(.533833+.251011*X))))
    return u1,u2,u3

@register_handler(63)
def u123_Eu(T):
    X = torch.log(5040/T)
    u1 = 8.+torch.exp(1.024374-X*(4.533653+X*(1.540805+X*(.827789+.286737*X))))
    u2 = 9.+torch.exp(1.92776+X*(-1.50646+X*(.379584+.05684*X)))
    u3 = torch.full_like(T,8)
    return u1,u2,u3

@register_handler(64)
def u123_Gd(T):
    X = torch.log(5040/T)
    u1 = 5.+torch.exp(4.009587+X*(-1.583513+X*(.800411+.388845*X)))
    u2 = 6.+torch.exp(4.362107-X*(1.208124+X*(-.074813+X*(.076453+.055475*X))))
    u3 = 5.+torch.exp(3.412951-X*(.50271+X*(.042489-4.017E-3*X)))
    return u1,u2,u3

@register_handler(65)
def u123_Tb(T):
    X = torch.log(5040/T)
    u1 = 16.+torch.exp(4.791661+X*(-1.249355+X*(.570094+.240203*X)))
    u2 = 15.+torch.exp(4.472549-X*(.295965+X*(5.88E-3+.131631*X)))
    u3 = u2
    return u1,u2,u3

@register_handler(66)
def u123_Dy(T):
    X = torch.log(5040/T)
    u1 = 17.+torch.exp(3.029646-X*(3.121036+X*(.086671-.216214*X)))
    u2 = 18.+torch.exp(3.465323-X*(1.27062+X*(-.382265+X*(.431447+.303575*X))))
    u3 = u2
    return u1,u2,u3

@register_handler(67)
def u123_Ho(T):
    X = torch.log(5040/T)
    u3 = 16.+torch.exp(1.610084-X*(2.373926+X*(.133139-.071196*X)))
    u2 = u3
    u1 = u3
    return u1,u2,u3

@register_handler(68)
def u123_Er(T):
    X = torch.log(5040/T)
    u1 = 13.+torch.exp(2.895648-X*(2.968603+X*(.561515+X*(.215267+.095813*X))))
    u2 = 14.+torch.exp(3.202542-X*(.852209+X*(-.226622+X*(.343738+.186042*X))))
    u3 = u2
    return u1,u2,u3

@register_handler(69)
def u123_Tm(T):
    X = torch.log(5040/T)
    u1 = 8.+torch.exp(1.021172-X*(4.94757+X*(1.081603+.034811*X)))
    u2 = 9.+torch.exp(2.173152+X*(-1.295327+X*(1.940395+.813303*X)))
    u3 = 8.+torch.exp(-.567398+X*(-3.383369+X*(.799911+.554397*X)))
    return u1,u2,u3

@register_handler(70)
def u123_Yb(T):
    X = torch.log(5040/T)
    u1 = 1.+torch.exp(-2.350549-X*(6.688837+X*(1.93869+.269237*X)))
    u2 = 2.+torch.exp(-3.047465-X*(7.390444+X*(2.355267+.44757*X)))
    u3 = 1.+torch.exp(-6.192056-X*(10.560552+X*(4.579385+.940171*X)))
    return u1,u2,u3

@register_handler(71)
def u123_Lu(T):
    X = torch.log(5040/T)
    u1 = 4.+EXP(1.537094+X*(-1.140264+X*(.608536+.193362*X)))
    u2 = torch.clamp(0.66+1.52E-4*T,min=1)
    u2 = torch.where(T>5250,-1.09+4.86E-4*T,u2)
    u3 = torch.full_like(T,5)
    return u1,u2,u3

@register_handler(72)
def u123_Hf(T):
    T3 = 1.e-3*T
    u1 = 4.1758+T3*(.407+T3*(.57862-T3*(.072887-T3*3.6848E-3)))
    u2 = -2.979+3.095E-3*T
    u3 = torch.full_like(T,30)
    return u1,u2,u3

@register_handler(73)
def u123_Ta(T):
    T3 = 1.e-3*T
    u1 = 3.0679+T3*(.81776+T3*(.34936+T3*(7.4861E-3+T3*3.0739E-4))) 
    u2 = 1.6834+T3*(2.0103+T3*(.56443-T3*(.031036-T3*8.9565E-4)))
    u3 = torch.full_like(T,15)
    return u1,u2,u3

@register_handler(74)
def u123_W(T):
    T3 = 1.e-3*T
    u1 = .3951+T3*(-.25057+T3*(1.4433+T3*(-.34373+T3*(.041924-T3*1.84E-3))))
    u1 = torch.where(T>1.2e4,23,u1)
    u2 = 1.055+T3*(1.0396+T3*(.3303-T3*(8.4971E-3-T3*5.5794E-4)))
    u3 = torch.full_like(T,20)
    return u1,u2,u3

@register_handler(75)
def u123_Re(T):
    T3 = 1.e-3*T
    u1 = 5.5671+T3*(.72721+T3*(-.42096+T3*(.09075-T3*3.9331E-3)))
    u1 = torch.where(T>1.2e4,29,u1)
    u2 = 6.5699+T3*(.59999+T3*(-.28532+T3*(.050724-T3*1.8544E-3)))
    u2 = torch.where(T>1.2e4,22,u2)
    u3 = torch.full_like(T,20)
    return u1,u2,u3

@register_handler(76)
def u123_Os(T):
    T3 = 1.e-3*T
    u1 = 8.6643+T3*(-.32516+T3*(.68181-T3*(.044252-T3*1.9975E-3)))
    u2 = 9.7086+T3*(-.3814+T3*(.65292-T3*(.064984-T3*2.8792E-3)))
    u3 = torch.full_like(T,10)
    return u1,u2,u3

@register_handler(77)
def u123_Ir(T):
    T3 = 1.e-3*T
    u1 = 11.07+T3*(-2.412+T3*(1.9388+T3*(-.34389+T3*(.033511-1.3376E-3*T3))))
    u1 = torch.where(T>1.2e4,30,u1)
    u2 = torch.full_like(T,15)
    u3 = torch.full_like(T,20)
    return u1,u2,u3

@register_handler(78)
def u123_Pt(T):
    T3 = 1.e-3*T
    u1 = 16.4+1.27E-3*T
    u2 = 6.5712+T3*(-1.0363+T3*(.57234-T3*(.061219-2.6878E-3*T3)))
    u3 = torch.full_like(T,15)
    return u1,u2,u3

@register_handler(79)
def u123_Au(T):
    T3 = 1.e-3*T
    u1 = 1.24+2.79E-4*T 
    u2 = 1.0546+T3*(-.040809+T3*(2.8439E-3+T3*1.6586E-3))
    u3 = torch.full_like(T,7)
    return u1,u2,u3

@register_handler(80)
def u123_Hg(T):
    u1 = torch.ones_like(T)
    u2 = torch.full_like(T,2)
    u3 = torch.clamp(.669+3.976E-5*T,min=1)
    return u1,u2,u3

@register_handler(81)
def u123_Tl(T):
    u1 = torch.clamp(0.63+3.35E-4*T,min=2)
    u2 = torch.ones_like(T)
    u3 = torch.full_like(T,2)
    return u1,u2,u3

@register_handler(82)
def u123_Pb(T):
    u1 = torch.clamp(0.42+2.35E-4*T,min=1)
    u1 = torch.where(T>6125,-1.2+5.E-4*T,u1)
    u2 = torch.clamp(1.72+7.9E-5*T,min=2)
    u3 = torch.ones_like(T)
    return u1,u2,u3

@register_handler(83)
def u123_Bi(T):
    u1 = 2.78+2.87E-4*T
    u2 = torch.clamp(.37+1.41E-4*T,min=1)
    u3 = torch.full_like(T,2.5)
    return u1,u2,u3

@register_handler(84)
def u123_Po(T):
    u1 = torch.full_like(T,5)
    u2 = torch.full_like(T,5)
    u3 = torch.full_like(T,4)
    return u1,u2,u3

@register_handler(85)
def u123_At(T):
    u1 = torch.full_like(T,4)
    u2 = torch.full_like(T,6)
    u3 = torch.full_like(T,6)
    return u1,u2,u3

@register_handler(86)
def u123_Rn(T):
    u1 = torch.full_like(T,1)
    u2 = torch.full_like(T,4)
    u3 = torch.full_like(T,6)
    return u1,u2,u3

@register_handler(87)
def u123_Fr(T):
    u1 = torch.full_like(T,2)
    u2 = torch.full_like(T,1)
    u3 = torch.full_like(T,4.5)
    return u1,u2,u3

@register_handler(88)
def u123_Ra(T):
    u1 = torch.full_like(T,1)
    u2 = torch.full_like(T,2)
    u3 = torch.full_like(T,1)
    return u1,u2,u3

@register_handler(89)
def u123_Ac(T):
    u1 = torch.full_like(T,6)
    u2 = torch.full_like(T,3)
    u3 = torch.full_like(T,7)
    return u1,u2,u3

@register_handler(90)
def u123_Th(T):
    u1 = torch.full_like(T,8)
    u2 = torch.full_like(T,8)
    u3 = torch.full_like(T,8)
    return u1,u2,u3

@register_handler(91)
def u123_Pa(T):
    u1 = torch.full_like(T,50)
    u2 = torch.full_like(T,50)
    u3 = torch.full_like(T,50)
    return u1,u2,u3

@register_handler(92)
def u123_U(T):
    u1 = torch.full_like(T,25)
    u2 = torch.full_like(T,25)
    u3 = torch.full_like(T,25)
    return u1,u2,u3


# hand out function
def u123(natom, T):
    if handler := handlers.get(natom):
        return handler(T)
    else:
        raise ValueError(f"Atomic number `{natom}` exceeding 92 is not supported")