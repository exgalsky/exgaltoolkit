"""
Cosmological parameters and validation.
"""
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

@dataclass
class CosmologicalParameters:
    """Cosmological parameters with validation."""
    
    # Primary parameters
    H0: float = 70.0          # Hubble constant [km/s/Mpc]
    Omega_m: float = 0.3      # Matter density parameter
    Omega_b: float = 0.049    # Baryon density parameter  
    Omega_lambda: float = 0.7 # Dark energy density parameter
    n_s: float = 0.965        # Scalar spectral index
    sigma_8: float = 0.81     # RMS matter fluctuation at 8 Mpc/h
    
    # Derived parameters
    h: Optional[float] = None         # Dimensionless Hubble parameter
    omega_m: Optional[float] = None   # Physical matter density
    omega_b: Optional[float] = None   # Physical baryon density
    omega_lambda: Optional[float] = None  # Physical dark energy density
    omega_k: Optional[float] = None   # Curvature density parameter
    
    def __post_init__(self):
        """Compute derived parameters and validate."""
        
        # Compute derived parameters
        # If h is explicitly provided, use it; otherwise derive from H0
        if self.h is None:
            self.h = self.H0 / 100.0
        else:
            # If h is provided, update H0 to be consistent
            self.H0 = self.h * 100.0
            
        self.omega_m = self.Omega_m * self.h**2
        self.omega_b = self.Omega_b * self.h**2
        self.omega_lambda = self.Omega_lambda
        self.omega_k = 1.0 - self.Omega_m - self.Omega_lambda
        
        # Basic validation
        if self.H0 <= 0:
            raise ValueError("H0 must be positive")
        if self.Omega_m <= 0:
            raise ValueError("Omega_m must be positive")
        if self.Omega_b <= 0:
            raise ValueError("Omega_b must be positive")
        if self.Omega_b > self.Omega_m:
            raise ValueError("Omega_b cannot be larger than Omega_m")
        if self.sigma_8 <= 0:
            raise ValueError("sigma_8 must be positive")

@dataclass
class PowerSpectrum:
    """Power spectrum configuration."""
    
    z_initial: float = 99.0    # Initial redshift
    k_min: float = 1e-4        # Minimum k [h/Mpc]
    k_max: float = 10.0        # Maximum k [h/Mpc]
    n_points: int = 1000       # Number of k points
    
    # Optional custom power spectrum
    k: Optional[jnp.ndarray] = None      # k array [h/Mpc]
    pofk: Optional[jnp.ndarray] = None   # P(k) array [(Mpc/h)^3]
    
    def __post_init__(self):
        """Validate power spectrum parameters."""
        if self.z_initial < 0:
            raise ValueError("z_initial must be non-negative")
        if self.k_min <= 0:
            raise ValueError("k_min must be positive")
        if self.k_max <= self.k_min:
            raise ValueError("k_max must be greater than k_min")
        if self.n_points <= 0:
            raise ValueError("n_points must be positive")
            
        # Validate custom power spectrum if provided
        if self.k is not None or self.pofk is not None:
            if self.k is None or self.pofk is None:
                raise ValueError("Both k and pofk must be provided for custom power spectrum")
            if len(self.k) != len(self.pofk):
                raise ValueError("k and pofk arrays must have same length")
            if jnp.any(self.k <= 0):
                raise ValueError("All k values must be positive")
            if jnp.any(self.pofk < 0):
                raise ValueError("All P(k) values must be non-negative")
