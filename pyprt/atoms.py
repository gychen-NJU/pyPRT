from .needs import *
from .phys import *
import pkg_resources

class atomic_config():
    def __init__(self, table_abu=None):
        if table_abu is None:
            table_abu = pkg_resources.resource_filename(
                'pyprt',  
                'data/table_abu.csv'
            )            
        self.table_abu = pd.read_csv(table_abu)
        self.df = pd.read_csv(table_abu)
        self.symbol = self.df['Symbol'].values
        self.abu = self.df['Abundance (12+log[x]/[H])'].values
        self.abu = np.power(10,np.array(self.abu)-12)
        self.wgt = self.df['Relative Mass'].values
        self.abusum = np.sum(self.abu)
        self.amw = np.sum(self.abu*self.wgt) # average molecular weights

    def __call__(self,natom):
        '''
        Calculate the atomic properties of natom
        ---------
        Input:
            natom: int -> atomic number

        Return:
            weight: float -> atomic weight
            abundance: float -> atomic abundance
            chi1: float -> first ionization potential
            chi2: float -> second ionization potential
        '''
        row = self.df.loc[natom-1].values.tolist()
        _,_,abundance,weight,chi1,chi2 = row
        abundance = np.power(10,np.array(abundance)-12)
        return weight,abundance,chi1,chi2

atomic_properties = atomic_config()
