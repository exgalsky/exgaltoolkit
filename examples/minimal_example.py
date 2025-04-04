# example for using a user-provided p(k) made with camb

import camb
import numpy as np
import jax.numpy as jnp
import exgaltoolkit.lpt as lpt
import exgaltoolkit.mockgen as mg

zics=100
camb_par = camb.set_params(H0=68)
camb_par.set_matter_power(redshifts=[zics], kmax=2.0)
camb_wsp = camb.get_results(camb_par)

def my_get_pspec():
    k, zlist, pk = camb_wsp.get_matter_power_spectrum(
                  minkh=1e-4, maxkh=1e2, npoints = 2000)
    return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}

# create Sky object
mocksky = mg.Sky(pspec=my_get_pspec(), N=128, seed=13579, Niter=1, icw=True)

# now write ICs
mocksky.run(laststep='writeics')
