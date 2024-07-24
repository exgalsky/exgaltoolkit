import scipy.constants as cons
from functools import partial
import jax

comov_lastscatter_Gpc = 13.8 # in Gpc

@partial(jax.jit, static_argnames=['cosmo_wsp', 'geometric_factor'])
def lensing_kernel_F(cosmo_wsp, geometric_factor, comov_q_i, redshift_i):
    comov_lastscatter = comov_lastscatter_Gpc * (cons.giga / cons.mega) # in Mpc
    return (geometric_factor * (3./2.) * cosmo_wsp.params['Omega_m'] * (cosmo_wsp.params['h'] * 100. * cons.kilo / cons.c )**2. * (1 + redshift_i) * (1. - (comov_q_i/comov_lastscatter)) / comov_q_i)