# example for using a user-provided p(k) made with camb

import camb
import numpy as np
import jax.numpy as jnp

zics=100
camb_par = camb.set_params()
camb_par.set_matter_power(redshifts=[zics], kmax=2.0)
camb_wsp = camb.get_results(camb_par)

def my_get_pspec():
    k, zlist, pk = camb_wsp.get_matter_power_spectrum(
                  minkh=1e-4, maxkh=1e2, npoints = 2000)
    return jnp.asarray(k), jnp.asarray(pk)

# create Sky object
mocksky = Sky(get_pspec=my_get_pspec)

# now write ICs
mocksky.run(laststep='writeics')

# THIS IS A SKETCH OF A POSSIBLE DEFAULT GET_PSPEC IMPLEMENTATION
# class CosmologyInterface:
#     '''CosmologyInterface'''    
#     def __init__(self, **kwargs):
#         self.h      = 0.7
#         self.omegam = 0.276
#         self.omegak = 0.0
#         self.omegal = 1 - self.omegam - self.omegak

#         def _get_pspec(self):
#             import numpy as np
#             import jax.numpy as jnp

#             # default power spectrum interface
#             from importlib.resources import files
#             pkfile = files("exgaltoolkit.data").joinpath("camb_40107036_matterpower.dat")
#             k, pk = np.loadtxt(pkfile,usecols=(0,1),unpack=True)
        
#             # create dictionary wrapper for power spectrum
#             self.pspec = {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}
#        self.get_pspec = kwargs.get('get_pspec', _get_pspec)
#         return
