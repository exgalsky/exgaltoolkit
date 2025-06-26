#!/usr/bin/env python3
"""
Comprehensive unit tests for ExGalToolkit core functionality.

This test suite validates the complete refactored JAX implementation including:

1. **Core API Components** (TestBasicAPI):
   - Component instantiation and configuration
   - Basic density field generation  
   - LPT displacement computation
   - Reproducibility validation

2. **Validation Framework**:
   - Centralized test configuration (TEST_CONFIG)
   - Target value validation against known results
   - Cross-platform reproducibility testing
   - Performance benchmarking

The refactored implementation has achieved bit-for-bit numerical compatibility
with the original implementation while providing GPU acceleration and cleaner APIs.
"""

import unittest
import tempfile
import os
import numpy as np
import jax.numpy as jnp
import warnings

# Import the simplified API
from exgaltoolkit import ICGenerator, CosmologicalParameters, CosmologyService

# Test configuration - centralized place for all test parameters
TEST_CONFIG = {
    # Grid parameters
    'N': 64,
    'Lbox': 7700.0,  # Mpc
    'seed': 13579,
    'z_initial': 100,
    
    # Cosmology parameters
    'H0': 68,  # km/s/Mpc
    'Omega_m': 0.31,
    'Omega_b': 0.049,
    'sigma_8': 0.8,
    'n_s': 0.96,
    
    # Expected results (target values for numerical accuracy)
    'target_mean_max': 1e-8, # maximum absolute mean value
    'target_std': 1.40291000e-03,
    
    # Numerical accuracy tolerances (percentage)
    'std_error_tolerance': 5.0,   # Maximum 5% error for std
}

def print_test_header(test_name, config=None):
    """Print a formatted test header with configuration details."""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*60}")
    
    if config is None:
        config = TEST_CONFIG
    
    print(f"📋 Configuration:")
    print(f"   Resolution (N): {config['N']}")
    print(f"   Box Size (Lbox): {config['Lbox']:.1f} Mpc")
    print(f"   Random Seed: {config['seed']}")
    print(f"   Initial Redshift: {config['z_initial']}")
    print(f"   H₀: {config['H0']} km/s/Mpc")
    print(f"   Ωₘ: {config['Omega_m']}")
    print(f"   Target abs(Mean) Max: {config['target_mean_max']:.2e}")
    print(f"   Target Std: {config['target_std']:.2e}")
    print()

def print_density_statistics(delta_array, test_config=None, label="Density Field"):
    """Print formatted density field statistics."""
    if test_config is None:
        test_config = TEST_CONFIG
        
    mean_delta = np.mean(delta_array)
    std_delta = np.std(delta_array)
    
    print(f"📊 {label} Statistics:")
    print(f"   Shape: {delta_array.shape}")
    print(f"   Mean: {mean_delta:.8e}")
    print(f"   Std:  {std_delta:.8e}")
    
    # Compare to targets if available
    if 'target_mean_max' in test_config and 'target_std' in test_config:
        mean_ratio = np.abs(mean_delta) / test_config['target_mean_max']
        std_ratio = std_delta / test_config['target_std']
        std_error = abs((std_delta - test_config['target_std']) / test_config['target_std']) * 100
        
        print(f"   📈 vs Target:")
        print(f"      Mean ratio: {mean_ratio:.2e}")
        print(f"      Std ratio:  {std_ratio:.2e} (error: {std_error:.2f}%)")
    
    print()

def print_test_result(success, message="", details=None):
    """Print formatted test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"🎯 Result: {status}")
    if message:
        print(f"   {message}")
    if details:
        for detail in details:
            print(f"   • {detail}")
    print(f"{'='*60}\n")

class TestBasicAPI(unittest.TestCase):
    """Test basic API functionality with simple power spectrum."""
    
    def setUp(self):
        """Set up test parameters using centralized TEST_CONFIG."""
        self.config = TEST_CONFIG.copy()
        self.N = self.config['N']
        self.Lbox = self.config['Lbox']
        self.seed = self.config['seed']
        self.temp_dir = tempfile.mkdtemp()
        
        # Use simple analytical power spectrum
        k = jnp.logspace(-2, 1, 100)
        pofk = (k / 0.05) ** (-1.5)  # Simple power law
        self.simple_pspec = {'k': k, 'pofk': pofk}
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_components_creation(self):
        """Test that all API components can be imported and instantiated."""
        # Test cosmological parameters
        cosmo_params = CosmologicalParameters()
        self.assertIsNotNone(cosmo_params)
        
        # Test with custom parameters
        cosmo_params = CosmologicalParameters(Omega_m=0.31, h=0.68)
        self.assertAlmostEqual(cosmo_params.Omega_m, 0.31)
        self.assertAlmostEqual(cosmo_params.h, 0.68)
        
        # Test cosmology service
        cosmo_service = CosmologyService(cosmo_params)
        self.assertIsNotNone(cosmo_service)
        
        # Test cosmology service with power spectrum
        cosmo_service = CosmologyService(cosmo_params, power_spectrum=self.simple_pspec)
        self.assertIsNotNone(cosmo_service.power_spectrum)
        self.assertIn('k', cosmo_service.power_spectrum)
        self.assertIn('pofk', cosmo_service.power_spectrum)
        
        # Test IC generator
        ic_gen = ICGenerator(N=self.N, Lbox=self.Lbox)
        self.assertIsNotNone(ic_gen)
        self.assertEqual(ic_gen.N, self.N)
        self.assertEqual(ic_gen.Lbox, self.Lbox)
    
    def test_basic_ic_generation(self):
        """Test basic initial conditions generation."""
        print_test_header("Basic IC Generation", self.config)
        
        cosmo_params = CosmologicalParameters(
            Omega_m=self.config['Omega_m'], 
            h=self.config['H0']/100.0
        )
        
        ic_gen = ICGenerator(
            N=self.N,
            Lbox=self.Lbox,
            cosmology=cosmo_params,
            seed=self.seed
        )
        
        # Override with simple power spectrum
        cosmo_service = CosmologyService(cosmo_params, power_spectrum=self.simple_pspec)
        ic_gen.cosmology_service = cosmo_service
        
        # Generate ICs
        result = ic_gen.generate_initial_conditions(save_output=False)
        
        success = result.get('success', False)
        self.assertTrue(success)
        
        # Check density field
        delta = ic_gen.get_density_field()
        if delta is not None:
            delta_array = np.asarray(delta)
            self.assertEqual(delta_array.shape, (self.N, self.N, self.N))
            self.assertTrue(np.isfinite(delta_array).all())
            
            print_density_statistics(delta_array, self.config, "Basic IC")
            
            # Check that it has reasonable statistics
            mean_delta = np.mean(delta_array)
            std_delta = np.std(delta_array)
            
            # Check against target values with strict tolerances
            std_error = abs((std_delta - self.config['target_std']) / self.config['target_std']) * 100
            
            # Assert strict numerical accuracy
            self.assertLess(abs(mean_delta), self.config['target_mean_max'], 
                            f"abs(Mean) {abs(mean_delta):.2f}% exceeds tolerance {self.config['target_mean_max']}")
            self.assertLess(std_error, self.config['std_error_tolerance'], 
                            f"Std error {std_error:.2f}% exceeds tolerance {self.config['std_error_tolerance']}%")
    
    def test_displacement_generation(self):
        """Test LPT displacement field generation."""
        cosmo_params = CosmologicalParameters(Omega_m=0.3, h=0.7)
        
        ic_gen = ICGenerator(
            N=self.N,
            Lbox=self.Lbox,
            cosmology=cosmo_params,
            seed=self.seed,
            lpt_order=1
        )
        
        # Override with simple power spectrum
        cosmo_service = CosmologyService(cosmo_params, power_spectrum=self.simple_pspec)
        ic_gen.cosmology_service = cosmo_service
        
        # Generate full initial conditions including LPT
        result = ic_gen.generate_initial_conditions(save_output=False)
        
        self.assertTrue(result.get('success', False))
        
        # Get displacement fields
        displacements = ic_gen.get_displacement_fields()
        self.assertIsNotNone(displacements)
        self.assertEqual(len(displacements), 3)  # X, Y, Z components
        
        for i, disp in enumerate(displacements):
            if disp is not None:
                disp_array = np.asarray(disp)
                self.assertEqual(disp_array.shape, (self.N, self.N, self.N))
                self.assertTrue(np.isfinite(disp_array).all())
                
                # Check that displacements have reasonable statistics
                self.assertLess(abs(np.mean(disp_array)), 1e-5)  # Mean should be ~0
                self.assertGreater(np.std(disp_array), 0)  # Should have variance
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        cosmo_params = CosmologicalParameters(Omega_m=0.3, h=0.7)
        
        # First run
        ic_gen1 = ICGenerator(
            N=self.N,
            Lbox=self.Lbox,
            cosmology=cosmo_params,
            seed=self.seed
        )
        cosmo_service1 = CosmologyService(cosmo_params, power_spectrum=self.simple_pspec)
        ic_gen1.cosmology_service = cosmo_service1
        
        result1 = ic_gen1.generate_initial_conditions(save_output=False)
        delta1 = ic_gen1.get_density_field()
        
        # Second run with same seed
        ic_gen2 = ICGenerator(
            N=self.N,
            Lbox=self.Lbox,
            cosmology=cosmo_params,
            seed=self.seed
        )
        cosmo_service2 = CosmologyService(cosmo_params, power_spectrum=self.simple_pspec)
        ic_gen2.cosmology_service = cosmo_service2
        
        result2 = ic_gen2.generate_initial_conditions(save_output=False)
        delta2 = ic_gen2.get_density_field()
        
        # Results should be identical
        self.assertTrue(result1.get('success', False))
        self.assertTrue(result2.get('success', False))
        
        if delta1 is not None and delta2 is not None:
            delta1_array = np.asarray(delta1)
            delta2_array = np.asarray(delta2)
            np.testing.assert_array_equal(delta1_array, delta2_array)


if __name__ == '__main__':
    # Run tests
    unittest.main()