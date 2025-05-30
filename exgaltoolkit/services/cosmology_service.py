"""
Cosmology and power spectrum services.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict
from ..core.config import CosmologicalParameters, PowerSpectrum
from ..core.data_models import GrowthFactors

class CosmologyService:
    """Service for cosmological calculations."""
    
    def __init__(self, parameters: CosmologicalParameters):
        self.parameters = parameters
        self._growth_cache: Optional[GrowthFactors] = None
    
    def compute_growth_factors(self, z_array: Optional[jnp.ndarray] = None) -> GrowthFactors:
        """Compute linear and second-order growth factors."""
        if z_array is None:
            z_array = jnp.linspace(0, 100, 1000)
        
        # Use existing logic from cosmo.py
        z = z_array
        
        # hubble parameter
        Hofz = 100 * self.parameters.h**2 * jnp.sqrt(
            self.parameters.omega_lambda + 
            self.parameters.omega_k * (1 + z)**2 + 
            self.parameters.omega_m * (1 + z)**3
        )

        # linear growth factor using w=-1 fitting formulae from Carroll, Press & Turner (1992)
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

        # f derivatives by autograd of growth factors
        D1ofa = lambda a: jnp.interp(a, 1/(1+z), D1)
        D2ofa = lambda a: jnp.interp(a, 1/(1+z), D2)

        f1ofz = jax.vmap(lambda z: jax.grad(D1ofa)(1/(1+z)) / D1ofa(1/(1+z)) / (1+z))
        f2ofz = jax.vmap(lambda z: jax.grad(D2ofa)(1/(1+z)) / D2ofa(1/(1+z)) / (1+z))

        f1 = f1ofz(z)
        f2 = f2ofz(z)

        return GrowthFactors(
            z=z,
            hubble=Hofz,
            d1=D1,
            d2=D2,
            f1=f1,
            f2=f2
        )
    
    def get_growth_factors(self, z_array: Optional[jnp.ndarray] = None) -> GrowthFactors:
        """Get growth factors (alias for compute_growth_factors for backward compatibility)."""
        return self.compute_growth_factors(z_array)
    
    def get_hubble_parameter(self, z: float) -> float:
        """Get Hubble parameter at redshift z."""
        return 100 * self.parameters.h * jnp.sqrt(
            self.parameters.omega_lambda + 
            self.parameters.omega_k * (1 + z)**2 + 
            self.parameters.omega_m * (1 + z)**3
        )

class PowerSpectrumService:
    """Service for power spectrum operations."""
    
    def __init__(self, power_spectrum: Optional[PowerSpectrum] = None):
        if power_spectrum is None:
            power_spectrum = self._get_default_power_spectrum()
        self.power_spectrum = power_spectrum
    
    def _get_default_power_spectrum(self) -> PowerSpectrum:
        """Get default power spectrum from data file."""
        from importlib.resources import files
        
        pkfile = files("exgaltoolkit.data").joinpath("camb_40107036_matterpower.dat")
        k, pk = np.loadtxt(pkfile, usecols=(0, 1), unpack=True)
        
        return PowerSpectrum(
            k=jnp.asarray(k),
            power=jnp.asarray(pk)
        )
    
    def get_transfer_function(self, k_grid: jnp.ndarray, d3k: float, N: int) -> jnp.ndarray:
        """Get transfer function for given k grid."""
        # If k and power arrays are not set, get default power spectrum
        if self.power_spectrum.k is None or self.power_spectrum.power is None:
            default_ps = self._get_default_power_spectrum()
            k_array = default_ps.k
            power_array = default_ps.power
        else:
            k_array = self.power_spectrum.k
            power_array = self.power_spectrum.power
            
        power_data = np.asarray([k_array, power_array])
        p_whitenoise = (2*np.pi)**3 / (d3k * N**3)  # white noise power spectrum
        transfer = power_data.copy()
        transfer[1] = (power_data[1] / p_whitenoise)**0.5  # transfer(k) = sqrt[P(k)/P_whitenoise]
        return jnp.asarray(transfer)
    
    @classmethod
    def from_camb(cls, camb_results, redshift: float = 0.0) -> 'PowerSpectrumService':
        """Create from CAMB results."""
        k, z_list, pk = camb_results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=1e2, npoints=2000
        )
        
        # Find closest redshift index
        z_idx = 0
        if len(z_list) > 1:
            z_idx = jnp.argmin(jnp.abs(jnp.array(z_list) - redshift))
        
        power_spectrum = PowerSpectrum(
            k=jnp.asarray(k),
            power=jnp.asarray(pk[z_idx, :])
        )
        
        return cls(power_spectrum)
    
    @classmethod
    def from_dict(cls, pspec_dict: Dict) -> 'PowerSpectrumService':
        """Create from dictionary (legacy interface)."""
        power_spectrum = PowerSpectrum(
            k=pspec_dict['k'],
            power=pspec_dict['pofk']
        )
        return cls(power_spectrum)
