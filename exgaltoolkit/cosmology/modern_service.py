"""
Modern JAX-Native Cosmology Service
Phase 2B: Clean Implementation without Legacy Dependencies
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, NamedTuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CosmologicalParameters:
    """Cosmological parameters container."""
    h: float = 0.7              # Hubble parameter H0 = 100*h km/s/Mpc
    Omega_m: float = 0.276       # Matter density parameter
    Omega_k: float = 0.0         # Curvature density parameter  
    Omega_Lambda: float = None   # Dark energy density parameter (computed if None)
    w: float = -1.0              # Dark energy equation of state
    
    def __post_init__(self):
        if self.Omega_Lambda is None:
            self.Omega_Lambda = 1.0 - self.Omega_m - self.Omega_k


class GrowthFactors(NamedTuple):
    """Growth factors and related quantities."""
    z: jnp.ndarray           # Redshift array
    H_z: jnp.ndarray         # Hubble parameter H(z)
    D1: jnp.ndarray          # Linear growth factor D1(z)
    D2: jnp.ndarray          # Second-order growth factor D2(z)  
    f1: jnp.ndarray          # Linear growth rate f1(z) = dlnD1/dlna
    f2: jnp.ndarray          # Second-order growth rate f2(z) = dlnD2/dlna


class ModernCosmologyService:
    """
    JAX-native cosmology service.
    
    This is a complete rewrite that:
    1. Uses pure JAX without legacy dependencies
    2. Preserves exact numerical behavior from legacy cosmo.py
    3. Implements clean, modern patterns
    4. Supports automatic differentiation for growth rates
    """
    
    def __init__(self, parameters: Optional[CosmologicalParameters] = None, 
                 pspec: Optional[Dict[str, jnp.ndarray]] = None):
        """
        Initialize cosmology service.
        
        Parameters:
        -----------
        parameters : CosmologicalParameters, optional
            Cosmological parameters. Uses default if None.
        pspec : dict, optional
            Power spectrum data {'k': k_array, 'pofk': P_k_array}.
            Loads default CAMB data if None.
        """
        self.parameters = parameters or CosmologicalParameters()
        self.pspec = pspec or self._load_default_power_spectrum()
        self._growth_factors = None
    
    def _load_default_power_spectrum(self) -> Dict[str, jnp.ndarray]:
        """Load default power spectrum from CAMB data file."""
        try:
            from importlib.resources import files
            pkfile = files("exgaltoolkit.data").joinpath("camb_40107036_matterpower.dat")
            k, pk = np.loadtxt(pkfile, usecols=(0, 1), unpack=True)
            return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}
        except Exception as e:
            # Fallback: Create simple power law for testing
            print(f"Warning: Could not load CAMB data ({e}), using fallback power spectrum")
            k = jnp.logspace(-3, 2, 1000)  # k from 0.001 to 100 h/Mpc
            pk = k**(-1.0) * jnp.exp(-k/10.0)  # Simple power law with cutoff
            return {'k': k, 'pofk': pk}
    
    def get_power_spectrum(self) -> Dict[str, jnp.ndarray]:
        """Get power spectrum data."""
        return self.pspec
    
    def get_hubble_parameter(self, z: float) -> float:
        """
        Get Hubble parameter H(z) in km/s/Mpc.
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        H_z : float
            Hubble parameter at redshift z
        """
        params = self.parameters
        H_z = (100 * params.h * jnp.sqrt(
            params.Omega_Lambda + 
            params.Omega_k * (1 + z)**2 + 
            params.Omega_m * (1 + z)**3
        ))
        return float(H_z)
    
    def get_growth_factors(self, z_max: float = 100.0, n_points: int = 1000) -> GrowthFactors:
        """
        Compute growth factors and growth rates.
        
        Uses fitting formulae from Carroll, Press & Turner (1992) for linear growth
        and Bernardeau et al. (2001) approximation for second-order growth.
        
        Parameters:
        -----------
        z_max : float
            Maximum redshift for calculation
        n_points : int
            Number of redshift points
            
        Returns:
        --------
        growth_factors : GrowthFactors
            Growth factors and rates as function of redshift
        """
        if self._growth_factors is None:
            self._growth_factors = self._compute_growth_factors(z_max, n_points)
        return self._growth_factors
    
    def _compute_growth_factors(self, z_max: float, n_points: int) -> GrowthFactors:
        """Internal method to compute growth factors."""
        params = self.parameters
        z = jnp.linspace(0, z_max, n_points)
        
        # Hubble parameter H(z)
        H_z = 100 * params.h * jnp.sqrt(
            params.Omega_Lambda * (1 + z)**(3 * (1 + params.w)) +
            params.Omega_k * (1 + z)**2 + 
            params.Omega_m * (1 + z)**3
        )
        
        # Density parameters as function of redshift
        x = 1 + z
        x2 = x * x  
        x3 = x * x * x
        x3w = x3**params.w
        
        # Matter density parameter Ω_m(z)
        omega_z = (params.Omega_m * x3 / 
                  (params.Omega_m * x3 + 
                   (1 - params.Omega_m - params.Omega_Lambda) * x2 + 
                   params.Omega_Lambda * x3 * x3w))
        
        # Dark energy density parameter Ω_Λ(z)  
        lambda_z = (params.Omega_Lambda * x3 * x3w /
                   (params.Omega_m * x3 + 
                    (1 - params.Omega_m - params.Omega_Lambda) * x2 + 
                    params.Omega_Lambda * x3 * x3w))
        
        # Linear growth factor (Carroll, Press & Turner 1992)
        g_z = (2.5 * omega_z / 
               (omega_z**(4./7.) - lambda_z + 
                (1 + omega_z/2) * (1 + lambda_z/70)))
        
        # Normalization at z=0
        omega_0 = params.Omega_m
        lambda_0 = params.Omega_Lambda  
        g_0 = (2.5 * omega_0 / 
               (omega_0**(4./7.) - lambda_0 + 
                (1 + omega_0/2) * (1 + lambda_0/70)))
        
        # Normalized linear growth factor
        D1 = (g_z / x) / g_0
        
        # Second-order growth factor (Bernardeau et al. 2001)
        D2 = -3./7. * params.Omega_m**(-1./143.) * D1**2
        
        # Growth rates using JAX automatic differentiation
        # f = d(ln D)/d(ln a) = -(1+z) * d(D)/dz / D
        def D1_of_a(a):
            z_interp = 1./a - 1.
            return jnp.interp(z_interp, z, D1)
        
        def D2_of_a(a):
            z_interp = 1./a - 1.
            return jnp.interp(z_interp, z, D2)
        
        # Compute growth rates via autodiff
        a_values = 1. / (1. + z)
        
        def compute_f1(z_val):
            a_val = 1. / (1. + z_val)
            dD1_da = jax.grad(D1_of_a)(a_val)
            D1_val = D1_of_a(a_val)
            f1_val = a_val * dD1_da / D1_val
            return f1_val
        
        def compute_f2(z_val):
            a_val = 1. / (1. + z_val) 
            dD2_da = jax.grad(D2_of_a)(a_val)
            D2_val = D2_of_a(a_val)
            f2_val = a_val * dD2_da / D2_val
            return f2_val
        
        # Vectorize the computation
        f1 = jax.vmap(compute_f1)(z)
        f2 = jax.vmap(compute_f2)(z)
        
        return GrowthFactors(
            z=z,
            H_z=H_z,
            D1=D1,
            D2=D2, 
            f1=f1,
            f2=f2
        )
    
    def get_linear_growth_factor(self, z: float) -> float:
        """Get linear growth factor D1(z) at specific redshift."""
        growth = self.get_growth_factors()
        D1_z = jnp.interp(z, growth.z, growth.D1)
        return float(D1_z)
    
    def get_linear_growth_rate(self, z: float) -> float:
        """Get linear growth rate f1(z) at specific redshift."""
        growth = self.get_growth_factors()
        f1_z = jnp.interp(z, growth.z, growth.f1)
        return float(f1_z)
    
    def get_second_order_growth_factor(self, z: float) -> float:
        """Get second-order growth factor D2(z) at specific redshift."""
        growth = self.get_growth_factors()
        D2_z = jnp.interp(z, growth.z, growth.D2)
        return float(D2_z)
    
    def get_second_order_growth_rate(self, z: float) -> float:
        """Get second-order growth rate f2(z) at specific redshift.""" 
        growth = self.get_growth_factors()
        f2_z = jnp.interp(z, growth.z, growth.f2)
        return float(f2_z)
    
    def evaluate_power_spectrum(self, k: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate power spectrum at given k values.
        
        Parameters:
        -----------
        k : jnp.ndarray
            Wavenumbers in h/Mpc
            
        Returns:
        --------
        P_k : jnp.ndarray
            Power spectrum values
        """
        return jnp.interp(k, self.pspec['k'], self.pspec['pofk'], 
                         left=0.0, right=0.0)
    
    def set_custom_power_spectrum(self, k: jnp.ndarray, pk: jnp.ndarray):
        """Set custom power spectrum data."""
        self.pspec = {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk)}
        # Clear cached growth factors since they might depend on power spectrum normalization
        self._growth_factors = None


# Legacy interface compatibility
def create_legacy_compatible_cosmology(h: float = 0.7, omegam: float = 0.276) -> ModernCosmologyService:
    """
    Create cosmology service with legacy-compatible interface.
    
    This maintains the same API as the legacy mockgen.CosmologyInterface
    for backward compatibility during transition.
    """
    params = CosmologicalParameters(h=h, Omega_m=omegam)
    return ModernCosmologyService(parameters=params)
