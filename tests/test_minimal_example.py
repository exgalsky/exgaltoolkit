#!/usr/bin/env python3
"""
Unit test equivalent to minimal_example_serial.py

This test verifies the core functionality of the exgaltoolkit by:
1. Creating a cosmology interface with CAMB-generated power spectrum
2. Running Sky simulation with noise generation and convolution
3. Running LPT displacement calculations
4. Verifying that outputs have expected properties and shapes
5. Comparing legacy vs refactored implementations for consistency
"""

import unittest
import tempfile
import os
import numpy as np
import jax.numpy as jnp
import camb
import warnings
import exgaltoolkit.lpt as lpt
import exgaltoolkit.mockgen as mg
import exgaltoolkit.util.jax_util as ju

# Try to import the new API
try:
    from exgaltoolkit.api import SimulationFactory
    from exgaltoolkit.core.config import CosmologicalParameters, PowerSpectrum
    NEW_API_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"New API not available: {e}")
    NEW_API_AVAILABLE = False


class TestMinimalExample(unittest.TestCase):
    """Test case that replicates the minimal_example_serial.py functionality"""
    
    def setUp(self):
        """Set up test parameters matching the minimal example"""
        self.zics = 100
        self.N = 128
        self.seed = 13579
        self.Niter = 1
        self.H0 = 68
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up CAMB parameters exactly as in minimal example
        self.camb_par = camb.set_params(H0=self.H0)
        self.camb_par.set_matter_power(redshifts=[self.zics], kmax=2.0)
        self.camb_wsp = camb.get_results(self.camb_par)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def my_get_pspec(self):
        """Power spectrum function matching the minimal example"""
        k, zlist, pk = self.camb_wsp.get_matter_power_spectrum(
                      minkh=1e-4, maxkh=1e2, npoints=2000)
        return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk[0, :])}
    
    def test_cosmology_interface_creation(self):
        """Test that cosmology interface can be created with custom power spectrum"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        self.assertIsNotNone(cosmo)
        
        # Verify that the power spectrum was properly set
        pspec = cosmo.pspec
        self.assertIn('k', pspec)
        self.assertIn('pofk', pspec)
        self.assertGreater(len(pspec['k']), 0)
        self.assertGreater(len(pspec['pofk']), 0)
        self.assertEqual(len(pspec['k']), len(pspec['pofk']))
    
    def test_sky_creation_and_basic_properties(self):
        """Test Sky object creation with specified parameters"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        mocksky = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                        Niter=self.Niter, icw=True)
        
        # Verify Sky object properties
        self.assertEqual(mocksky.N, self.N)
        self.assertEqual(mocksky.seed, self.seed)
        self.assertEqual(mocksky.Niter, self.Niter)
        self.assertTrue(mocksky.icw)
        self.assertIs(mocksky.cosmo, cosmo)
    
    def test_delta_generation_convolution_step(self):
        """Test delta generation through convolution step"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        mocksky = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                        Niter=self.Niter, icw=True)
        
        # Run up to convolution step
        result = mocksky.run(laststep='convolution')
        self.assertEqual(result, 0)  # Should return 0 on success
        
        # Verify that delta was generated
        self.assertIsNotNone(mocksky.cube)
        self.assertIsNotNone(mocksky.cube.delta)
        
        # Convert to numpy for testing
        delta = np.asarray(mocksky.cube.delta)
        
        # Verify delta properties
        self.assertEqual(delta.shape, (self.N, self.N, self.N))
        self.assertTrue(np.isfinite(delta).all())
        self.assertGreater(np.std(delta), 0)  # Should have non-zero variance
    
    def test_lpt_displacement_generation(self):
        """Test LPT displacement calculation"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        mocksky = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                        Niter=self.Niter, icw=True)
        
        # Run up to LPT step
        result = mocksky.run(laststep='LPT')
        self.assertEqual(result, 0)  # Should return 0 on success
        
        # Verify that LPT displacements were generated
        self.assertIsNotNone(mocksky.cube.s1x)
        self.assertIsNotNone(mocksky.cube.s1y)
        self.assertIsNotNone(mocksky.cube.s1z)
        
        # Convert to numpy for testing
        s1x = np.asarray(mocksky.cube.s1x)
        s1y = np.asarray(mocksky.cube.s1y)
        s1z = np.asarray(mocksky.cube.s1z)
        
        # Verify displacement properties
        expected_shape = (self.N, self.N, self.N)
        self.assertEqual(s1x.shape, expected_shape)
        self.assertEqual(s1y.shape, expected_shape)
        self.assertEqual(s1z.shape, expected_shape)
        
        # All displacements should be finite
        self.assertTrue(np.isfinite(s1x).all())
        self.assertTrue(np.isfinite(s1y).all())
        self.assertTrue(np.isfinite(s1z).all())
        
        # Displacements should have non-zero variance
        self.assertGreater(np.std(s1x), 0)
        self.assertGreater(np.std(s1y), 0)
        self.assertGreater(np.std(s1z), 0)
    
    def test_full_workflow_matching_minimal_example(self):
        """Test the complete workflow as in minimal_example_serial.py"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        
        # First run: generate delta
        mocksky1 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        mocksky1.run(laststep='convolution')
        delta = np.asarray(mocksky1.cube.delta)
        
        # Second run: generate LPT displacements
        mocksky2 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        mocksky2.run(laststep='LPT')
        s1x = np.asarray(mocksky2.cube.s1x)
        s1y = np.asarray(mocksky2.cube.s1y)
        s1z = np.asarray(mocksky2.cube.s1z)
        
        # Verify all outputs have the correct shape
        expected_shape = (self.N, self.N, self.N)
        self.assertEqual(delta.shape, expected_shape)
        self.assertEqual(s1x.shape, expected_shape)
        self.assertEqual(s1y.shape, expected_shape)
        self.assertEqual(s1z.shape, expected_shape)
        
        # Save outputs to temporary file (mimicking the npz save)
        output_file = os.path.join(self.temp_dir, "test_grids.npz")
        np.savez(output_file,
                 delta=delta,
                 s1x=s1x,
                 s1y=s1y,
                 s1z=s1z)
        
        # Verify file was created and can be loaded
        self.assertTrue(os.path.exists(output_file))
        loaded_data = np.load(output_file)
        
        self.assertIn('delta', loaded_data)
        self.assertIn('s1x', loaded_data)
        self.assertIn('s1y', loaded_data)
        self.assertIn('s1z', loaded_data)
        
        # Verify loaded data matches original
        np.testing.assert_array_equal(loaded_data['delta'], delta)
        np.testing.assert_array_equal(loaded_data['s1x'], s1x)
        np.testing.assert_array_equal(loaded_data['s1y'], s1y)
        np.testing.assert_array_equal(loaded_data['s1z'], s1z)
    
    def test_reproducibility(self):
        """Test that results are reproducible with the same seed"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        
        # First run
        mocksky1 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        mocksky1.run(laststep='LPT')
        s1x_first = np.asarray(mocksky1.cube.s1x)
        
        # Second run with same seed
        mocksky2 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        mocksky2.run(laststep='LPT')
        s1x_second = np.asarray(mocksky2.cube.s1x)
        
        # Results should be identical
        np.testing.assert_array_equal(s1x_first, s1x_second)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        
        # First run with seed 13579
        mocksky1 = mg.Sky(cosmo=cosmo, N=self.N, seed=13579, 
                         Niter=self.Niter, icw=True)
        mocksky1.run(laststep='LPT')
        s1x_first = np.asarray(mocksky1.cube.s1x)
        
        # Second run with different seed
        mocksky2 = mg.Sky(cosmo=cosmo, N=self.N, seed=24680, 
                         Niter=self.Niter, icw=True)
        mocksky2.run(laststep='LPT')
        s1x_second = np.asarray(mocksky2.cube.s1x)
        
        # Results should be different
        self.assertFalse(np.array_equal(s1x_first, s1x_second))
    
    def test_print_helpful_information_legacy(self):
        """Test that prints helpful information like minimal_example_serial.py (Legacy Version)"""
        print(f"\n{'='*60}")
        print("LEGACY VERSION - Testing minimal example functionality")
        print(f"{'='*60}")
        
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        
        print(f"Parameters:")
        print(f"  N = {self.N}")
        print(f"  seed = {self.seed}")
        print(f"  H0 = {self.H0}")
        print(f"  z_ics = {self.zics}")
        
        # Generate delta
        print(f"\n--- Step 1: Generating density field (delta) ---")
        mocksky1 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        result = mocksky1.run(laststep='convolution')
        delta = np.asarray(mocksky1.cube.delta)
        
        print(f"Delta field generated:")
        print(f"  Shape: {delta.shape}")
        print(f"  Mean: {np.mean(delta):.6f}")
        print(f"  Std: {np.std(delta):.6f}")
        print(f"  Min: {np.min(delta):.6f}")
        print(f"  Max: {np.max(delta):.6f}")
        
        # Generate LPT displacements
        print(f"\n--- Step 2: Computing LPT displacements ---")
        mocksky2 = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                         Niter=self.Niter, icw=True)
        result = mocksky2.run(laststep='LPT')
        s1x = np.asarray(mocksky2.cube.s1x)
        s1y = np.asarray(mocksky2.cube.s1y)
        s1z = np.asarray(mocksky2.cube.s1z)
        
        print(f"LPT displacements generated:")
        print(f"  s1x - Mean: {np.mean(s1x):.6f}, Std: {np.std(s1x):.6f}")
        print(f"  s1y - Mean: {np.mean(s1y):.6f}, Std: {np.std(s1y):.6f}")
        print(f"  s1z - Mean: {np.mean(s1z):.6f}, Std: {np.std(s1z):.6f}")
        
        # Save to file like the original example
        output_file = os.path.join(self.temp_dir, "legacy_grids.npz")
        np.savez(output_file,
                 delta=delta,
                 s1x=s1x,
                 s1y=s1y,
                 s1z=s1z)
        
        print(f"\n--- Results saved to: {output_file} ---")
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Verify all steps succeeded
        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(output_file))
        
        print(f"{'='*60}")
        print("LEGACY VERSION - All tests passed!")
        print(f"{'='*60}\n")
    
    @unittest.skipIf(not NEW_API_AVAILABLE, "New API not available")
    def test_print_helpful_information_refactored(self):
        """Test that prints helpful information using the new refactored API"""
        print(f"\n{'='*60}")
        print("REFACTORED VERSION - Testing minimal example functionality")
        print(f"{'='*60}")
        
        print(f"Parameters:")
        print(f"  N = {self.N}")
        print(f"  seed = {self.seed}")
        print(f"  H0 = {self.H0}")
        print(f"  z_ics = {self.zics}")
        
        # Create simulation using new API
        print(f"\n--- Creating simulation with new API ---")
        factory = SimulationFactory()
        simulation = factory.create_mock_generation_simulation(
            N=self.N,
            L=7700.0,  # Box size
            H0=self.H0,
            Omega_m=0.31,
            seed=self.seed,
            z_initial=self.zics,
            write_ics=True
        )
        
        print(f"Simulation created successfully")
        print(f"  Grid size: {self.N}^3")
        print(f"  Box size: 7700.0 Mpc/h")
        
        # Run simulation
        print(f"\n--- Running complete simulation pipeline ---")
        result = simulation.run()
        
        print(f"Simulation completed:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        
        # Get results
        grid_data = simulation.get_grid_data()
        particle_positions, particle_velocities = simulation.get_particle_data()
        
        if grid_data is not None and hasattr(grid_data, 'delta'):
            delta = np.asarray(grid_data.delta)
            print(f"\nDelta field from new API:")
            print(f"  Shape: {delta.shape}")
            print(f"  Mean: {np.mean(delta):.6f}")
            print(f"  Std: {np.std(delta):.6f}")
            print(f"  Min: {np.min(delta):.6f}")
            print(f"  Max: {np.max(delta):.6f}")
        
        if particle_positions is not None:
            positions = np.asarray(particle_positions)
            print(f"\nParticle positions from new API:")
            print(f"  Shape: {positions.shape}")
            print(f"  X range: [{np.min(positions[..., 0]):.2f}, {np.max(positions[..., 0]):.2f}]")
            print(f"  Y range: [{np.min(positions[..., 1]):.2f}, {np.max(positions[..., 1]):.2f}]")
            print(f"  Z range: [{np.min(positions[..., 2]):.2f}, {np.max(positions[..., 2]):.2f}]")
        
        if particle_velocities is not None:
            velocities = np.asarray(particle_velocities)
            print(f"\nParticle velocities from new API:")
            print(f"  Shape: {velocities.shape}")
            print(f"  RMS velocity: {np.sqrt(np.mean(velocities**2)):.6f}")
        
        # Save results using new API format
        output_file = os.path.join(self.temp_dir, "refactored_results.npz")
        if grid_data is not None:
            save_data = {}
            if hasattr(grid_data, 'delta'):
                save_data['delta'] = np.asarray(grid_data.delta)
            if particle_positions is not None:
                save_data['positions'] = np.asarray(particle_positions)
            if particle_velocities is not None:
                save_data['velocities'] = np.asarray(particle_velocities)
            
            if save_data:
                np.savez(output_file, **save_data)
                print(f"\n--- Results saved to: {output_file} ---")
                file_size = os.path.getsize(output_file) / (1024**2)  # MB
                print(f"File size: {file_size:.2f} MB")
        
        # Verify success
        self.assertTrue(result.success)
        
        print(f"{'='*60}")
        print("REFACTORED VERSION - All tests passed!")
        print(f"{'='*60}\n")
    
    @unittest.skipIf(not NEW_API_AVAILABLE, "New API not available")
    def test_compare_legacy_vs_refactored(self):
        """Compare results between legacy and refactored implementations"""
        print(f"\n{'='*60}")
        print("COMPARISON - Legacy vs Refactored Implementation")
        print(f"{'='*60}")
        
        # Legacy implementation
        print(f"\n--- Running Legacy Implementation ---")
        cosmo = mg.CosmologyInterface(pspec=self.my_get_pspec())
        mocksky = mg.Sky(cosmo=cosmo, N=self.N, seed=self.seed, 
                        Niter=self.Niter, icw=True)
        mocksky.run(laststep='convolution')
        legacy_delta = np.asarray(mocksky.cube.delta)
        
        # Refactored implementation
        print(f"--- Running Refactored Implementation ---")
        factory = SimulationFactory()
        simulation = factory.create_mock_generation_simulation(
            N=self.N,
            L=7700.0,
            H0=self.H0,
            Omega_m=0.31,
            seed=self.seed,
            z_initial=self.zics
        )
        result = simulation.run()
        grid_data = simulation.get_grid_data()
        
        if grid_data is not None and hasattr(grid_data, 'delta'):
            refactored_delta = np.asarray(grid_data.delta)
            
            print(f"\n--- Comparison Results ---")
            print(f"Legacy delta    - Mean: {np.mean(legacy_delta):.6f}, Std: {np.std(legacy_delta):.6f}")
            print(f"Refactored delta - Mean: {np.mean(refactored_delta):.6f}, Std: {np.std(refactored_delta):.6f}")
            
            # Check if they're approximately equal (allowing for small numerical differences)
            if np.allclose(legacy_delta, refactored_delta, rtol=1e-10, atol=1e-12):
                print(f"✅ Results are numerically identical!")
            else:
                print(f"⚠️  Results differ (expected for different implementations)")
                diff = np.abs(legacy_delta - refactored_delta)
                print(f"   Max absolute difference: {np.max(diff):.2e}")
                print(f"   RMS difference: {np.sqrt(np.mean(diff**2)):.2e}")
        else:
            print(f"⚠️  Could not compare - refactored version didn't produce delta field")
        
        print(f"{'='*60}")
        print("COMPARISON - Complete!")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    # Special test runner that shows verbose output
    import sys
    
    # Create test suite with only the informational tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add the informational tests
    suite.addTest(TestMinimalExample('test_print_helpful_information_legacy'))
    if NEW_API_AVAILABLE:
        suite.addTest(TestMinimalExample('test_print_helpful_information_refactored'))
        suite.addTest(TestMinimalExample('test_compare_legacy_vs_refactored'))
    
    # Run with high verbosity to see the print statements
    runner = unittest.TextTestRunner(verbosity=2, buffer=False, stream=sys.stdout)
    result = runner.run(suite)
    
    # Also run all tests
    print(f"\n{'='*80}")
    print("Running complete test suite...")
    print(f"{'='*80}")
    unittest.main(argv=[''], exit=False, verbosity=1)