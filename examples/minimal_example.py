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

# Create cosmology interface first
cosmo = mg.CosmologyInterface(pspec=my_get_pspec())

# create Sky object with the cosmology interface
mocksky = mg.Sky(cosmo=cosmo, N=128, seed=13579, Niter=1, icw=True)

# now write delta
mocksky.run(laststep='convolution')
delta=np.asarray(mocksky.cube.delta)

# create another Sky object with the same cosmology interface
mocksky = mg.Sky(cosmo=cosmo, N=128, seed=13579, Niter=1, icw=True)

# now write s1x
mocksky.run(laststep='LPT')
np.savez("./output/grids",
         delta=delta,
         s1x=np.asarray(mocksky.cube.s1x),
         s1y=np.asarray(mocksky.cube.s1y),
         s1z=np.asarray(mocksky.cube.s1z))
