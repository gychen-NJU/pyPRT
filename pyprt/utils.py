from typing import Any
from unittest import skip

from .needs import *
import pkg_resources

def lines_option(acquire=False):
    """
    Show the optional lines
    """
    lines_file = pkg_resources.resource_filename(
        'pyprt',
        'data/lines.json'
    )
    with open(lines_file) as f:
        lines = json.load(f)
    if acquire:
        return lines
    else:
        option = {iline:list[Any](lines[iline].keys()) for iline in lines.keys()}
        print(json.dumps(option, indent=2, ensure_ascii=False))

def load_lines(lines_usr):
    lines_sys = lines_option(acquire=True)
    lines = dict()
    for iline in lines_usr:
        if iline not in lines_sys.keys():
            raise KeyError(f"{iline} is not supported now. Please choose in {tuple(lines_sys.keys())}")
        else:
            for iwav in lines_usr[iline]:
                if iwav not in lines_sys[iline]:
                    raise KeyError(f"{iwav} is not supported now. Please choose in {tuple(lines_sys[iline].keys())}")
                else:
                    lines[f'{iline}_{iwav}'] = lines_sys[iline][iwav]
    return lines

def load_initial_guess(model='Umbral_big_spot'):
    path = pkg_resources.resource_filename(
        'pyprt',
        f'data/model_atmosphere'
    )
    files = sorted(glob.glob(os.path.join(path,"*.txt")))
    model_options = {os.path.basename(ifile).split(".")[0] for ifile in files}
    if model not in model_options:
        raise ValueError(f"{model} is not supported now. Please choose in {tuple(model_options)}")
    else:
        data = np.loadtxt(os.path.join(path,f"{model}.txt"), skiprows=2)
        initial_guess = dict(
            ltau        = data[:,0],
            T           = data[:,1],
            Pe          = data[:,2],
            vmic        = data[:,3],
            Bmag        = data[:,4],
            vLos        = data[:,5],
            inclination = data[:,6],
            azimuth     = data[:,7],
            Pg          = data[:,9],
            rhog        = data[:,10],
        )
        return initial_guess
