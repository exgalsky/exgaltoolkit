"""
Legacy CosmologyInterface wrapper for backward compatibility.
"""
from typing import Optional, Dict, Any
import warnings


class CosmologyInterface:
    """Legacy CosmologyInterface wrapper for backward compatibility."""
    
    def __init__(self, **kwargs):
        """Initialize cosmology interface with legacy parameters."""
        self._kwargs = kwargs
        
        # Initialize cosmology service
        try:
            from ..core.config import CosmologicalParameters, PowerSpectrum
            from ..services.cosmology_service import CosmologyService, PowerSpectrumService
            
            # Create configuration from kwargs
            cosmo_params = CosmologicalParameters.from_legacy_kwargs(**kwargs)
            power_spectrum = PowerSpectrum.from_legacy_kwargs(**kwargs)
            
            # Initialize services
            self._cosmology_service = CosmologyService(cosmo_params)
            self._power_spectrum_service = PowerSpectrumService(
                self._cosmology_service, power_spectrum
            )
            
        except Exception as e:
            warnings.warn(f"New cosmology interface failed, will fall back to legacy: {e}")
            self._cosmology_service = None
            self._power_spectrum_service = None
    
    def get_pspec(self):
        """Get power spectrum."""
        if self._power_spectrum_service:
            try:
                return self._power_spectrum_service.get_power_spectrum()
            except Exception as e:
                warnings.warn(f"New power spectrum failed, falling back to legacy: {e}")
        
        # Fall back to original implementation
        from ..mockgen.cosmo import CosmologyInterface as OriginalCosmo
        
        if not hasattr(self, '_original_cosmo'):
            self._original_cosmo = OriginalCosmo(**self._kwargs)
            
        return self._original_cosmo.get_pspec()
    
    def get_growth(self):
        """Get growth factors."""
        if self._cosmology_service:
            try:
                return self._cosmology_service.get_growth_factors()
            except Exception as e:
                warnings.warn(f"New growth factors failed, falling back to legacy: {e}")
        
        # Fall back to original implementation
        from ..mockgen.cosmo import CosmologyInterface as OriginalCosmo
        
        if not hasattr(self, '_original_cosmo'):
            self._original_cosmo = OriginalCosmo(**self._kwargs)
            
        return self._original_cosmo.get_growth()
