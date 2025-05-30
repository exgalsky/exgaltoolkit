#!/usr/bin/env python3
"""
Unit test equivalent to minimal_example_serial.py

This test verifies the core functionality of the exgaltoolkit by:
1. Creating a cosmology interface with CAMB-generated power spectrum
2. Running Sky simulation with noise generation and convolution
3. Running LPT displacement calculations
4. Verifying that outputs have expected properties and shapes
"""

import unittest
import tempfile
import os
import numpy as np
import jax.numpy as jnp
import camb
import exgaltoolkit.lpt as lpt
import exgaltoolkit.mockgen as mg
import exgaltoolkit.util.jax_util as ju


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


if __name__ == '__main__':
    unittest.main()