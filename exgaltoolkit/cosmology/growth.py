"""
Growth factors and cosmology service.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from .parameters import CosmologicalParameters, PowerSpectrum

@dataclass
class GrowthFactors:
    """Container for cosmological growth factors."""
    z: jnp.ndarray          # Redshift array
    hubble: jnp.ndarray     # Hubble parameter H(z)
    d1: jnp.ndarray         # Linear growth factor D1(z)
    d2: jnp.ndarray         # Second-order growth factor D2(z)
    f1: jnp.ndarray         # Linear growth rate f1(z)
    f2: jnp.ndarray         # Second-order growth rate f2(z)

class CosmologyService:
    """Service for cosmological calculations."""
    
    def __init__(self, parameters: CosmologicalParameters, power_spectrum: Optional[dict] = None):
        self.parameters = parameters
        self._growth_cache: Optional[GrowthFactors] = None
        
        # Set up power spectrum (for compatibility with existing cube interface)
        if power_spectrum is not None:
            self.pspec = power_spectrum
        else:
            self.pspec = self._get_default_power_spectrum()
        
        # Legacy compatibility attributes
        self.h = parameters.h
        self.omegam = parameters.omega_m
        self.omegak = parameters.omega_k
        self.omegal = parameters.omega_lambda
    
    def _get_default_power_spectrum(self) -> dict:
        """Get default power spectrum from data file."""
        try:
            import numpy as np
            from importlib.resources import files
            
            pkfile = files("exgaltoolkit.data").joinpath("camb_40107036_matterpower.dat")
            k, pk = np.loadtxt(pkfile, usecols=(0, 1), unpack=True)
            
            return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}
        except Exception:
            # Fallback minimal power spectrum
            k = jnp.logspace(-3, 1, 100)
            pk = jnp.ones_like(k) * 1e-3  # Minimal flat power spectrum
            return {'k': k, 'pofk': pk}
    
    def compute_growth_factors(self, z_array: Optional[jnp.ndarray] = None) -> GrowthFactors:
        """Compute linear and second-order growth factors."""
        if z_array is None:
            z_array = jnp.logspace(-2, 2, 100)  # Default z range
        
        z = z_array
        
        # Hubble parameter H(z)
        Hofz = 100 * self.parameters.h**2 * jnp.sqrt(
            self.parameters.omega_lambda + 
            self.parameters.omega_k * (1 + z)**2 + 
            self.parameters.omega_m * (1 + z)**3
        )

        # Linear growth factor using w=-1 fitting formulae from Carroll, Press & Turner (1992)
        w = -1
        x = 1 + z
        x2 = x * x
        x3 = x * x * x
        x3w = x3**w

        omega = self.parameters.omega_m * x3 / (
            self.parameters.omega_m * x3 + 
            (1 - self.parameters.omega_m - self.parameters.omega_lambda) * x2 + 
            self.parameters.omega_lambda
        )
        Lambda = self.parameters.omega_lambda * x3 * x3w / (
            self.parameters.omega_m * x3 + 
            (1 - self.parameters.omega_m - self.parameters.omega_lambda) * x2 + 
            self.parameters.omega_lambda * x3 * x3w
        )

        g = 2.5 * omega / (omega**(4./7.) - Lambda + (1 + omega/2) * (1 + Lambda/70))
        g0 = 2.5 * self.parameters.omega_m / (
            self.parameters.omega_m**(4./7.) - self.parameters.omega_lambda + 
            (1 + self.parameters.omega_m/2) * (1 + self.parameters.omega_lambda/70)
        )
        D1 = (g/x) / g0

        # 2nd order growth factor approximation from Bernardeau et al. (2001)
        D2 = -3/7 * self.parameters.omega_m**(-1/143) * D1**2

        # Growth rates using JAX autodiff
        D1ofa = lambda a: jnp.interp(a, 1/(1+z), D1)
        D2ofa = lambda a: jnp.interp(a, 1/(1+z), D2)

        f1ofz = jax.vmap(lambda z_val: jax.grad(D1ofa)(1/(1+z_val)) / D1ofa(1/(1+z_val)) / (1+z_val))
        f2ofz = jax.vmap(lambda z_val: jax.grad(D2ofa)(1/(1+z_val)) / D2ofa(1/(1+z_val)) / (1+z_val))

        f1 = f1ofz(z)
        f2 = f2ofz(z)

        return GrowthFactors(
            z=z, hubble=Hofz, d1=D1, d2=D2, f1=f1, f2=f2
        )
    
    def get_growth_factors(self, z_array: Optional[jnp.ndarray] = None) -> GrowthFactors:
        """Get growth factors (cached)."""
        if self._growth_cache is None:
            self._growth_cache = self.compute_growth_factors(z_array)
        return self._growth_cache
    
    def get_hubble_parameter(self, z: float) -> float:
        """Get Hubble parameter at redshift z."""
        return 100 * self.parameters.h**2 * jnp.sqrt(
            self.parameters.omega_lambda + 
            self.parameters.omega_k * (1 + z)**2 + 
            self.parameters.omega_m * (1 + z)**3
        )
    
    def get_growth(self):
        """Get growth factors in legacy format (for compatibility with existing code)."""
        growth_factors = self.get_growth_factors()
        # Return in the format expected by legacy code: [z, H(z), D1, D2, f1, f2]
        self.growth = jnp.asarray([
            growth_factors.z,
            growth_factors.hubble,
            growth_factors.d1,
            growth_factors.d2,
            growth_factors.f1,
            growth_factors.f2
        ])
        return self.growth
