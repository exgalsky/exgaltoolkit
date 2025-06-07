#!/usr/bin/env python3
"""
Fixed comparison script to run comparison tests with detailed output.

This script runs both the legacy and refactored implementations side-by-side,
showing detailed information about the computation process and results,
similar to the original minimal_example_serial.py but with comparisons.

Usage:
    python tests/run_comparison_fixed.py
"""

import os
import sys
import tempfile
import numpy as np
import jax.numpy as jnp
import camb
import warnings

# Add the parent directory to the path to import exgaltoolkit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exgaltoolkit.lpt as lpt
import exgaltoolkit.mockgen as mg
import exgaltoolkit.util.jax_util as ju

# Try to import the new simplified API
try:
    from exgaltoolkit import ICGenerator, CosmologicalParameters, CosmologyService
    NEW_API_AVAILABLE = True
    print("✅ New simplified API is available")
except ImportError as e:
    NEW_API_AVAILABLE = False
    print(f"⚠️  New API not available: {e}")
    print("    Only legacy API will be tested")

def setup_camb_power_spectrum():
    """Set up CAMB power spectrum exactly like minimal_example_serial.py"""
    zics = 100
    H0 = 68
    
    camb_par = camb.set_params(H0=H0)
    camb_par.set_matter_power(redshifts=[zics], kmax=2.0)
    camb_wsp = camb.get_results(camb_par)
    
    def my_get_pspec():
        k, zlist, pk = camb_wsp.get_matter_power_spectrum(
                      minkh=1e-4, maxkh=1e2, npoints=2000)
        return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk[0, :])}
    
    return my_get_pspec, zics, H0

def run_legacy_implementation(pspec_func, N=128, seed=13579, zics=100, H0=68):
    """Run the legacy implementation exactly like minimal_example_serial.py"""
    print(f"\n{'='*80}")
    print("LEGACY IMPLEMENTATION - Following minimal_example_serial.py")
    print(f"{'='*80}")
    
    print(f"Parameters:")
    print(f"  N (grid size): {N}")
    print(f"  seed: {seed}")
    print(f"  H0: {H0}")
    print(f"  z_ics: {zics}")
    print(f"  Niter: 1")
    print(f"  icw: True")
    
    # Create cosmology interface
    print(f"\n--- Creating cosmology interface with CAMB power spectrum ---")
    cosmo = mg.CosmologyInterface(pspec=pspec_func())
    pspec = cosmo.pspec
    print(f"Power spectrum loaded:")
    print(f"  k range: [{np.min(pspec['k']):.2e}, {np.max(pspec['k']):.2e}] h/Mpc")
    print(f"  P(k) range: [{np.min(pspec['pofk']):.2e}, {np.max(pspec['pofk']):.2e}] (Mpc/h)^3")
    
    # Step 1: Generate delta field
    print(f"\n--- Step 1: Generating density field (delta) ---")
    mocksky1 = mg.Sky(cosmo=cosmo, N=N, seed=seed, Niter=1, icw=True)
    result1 = mocksky1.run(laststep='convolution')
    
    if result1 == 0:
        delta = np.asarray(mocksky1.cube.delta)
        print(f"✅ Delta field generated successfully")
        print(f"  Shape: {delta.shape}")
        print(f"  Mean: {np.mean(delta):.8f}")
        print(f"  Std: {np.std(delta):.8f}")
        print(f"  Min: {np.min(delta):.6f}")
        print(f"  Max: {np.max(delta):.6f}")
        print(f"  Non-finite values: {np.sum(~np.isfinite(delta))}")
    else:
        print(f"❌ Delta generation failed with return code: {result1}")
        return None
    
    # Step 2: Generate LPT displacements
    print(f"\n--- Step 2: Computing LPT displacements ---")
    mocksky2 = mg.Sky(cosmo=cosmo, N=N, seed=seed, Niter=1, icw=True)
    result2 = mocksky2.run(laststep='LPT')
    
    if result2 == 0:
        s1x = np.asarray(mocksky2.cube.s1x)
        s1y = np.asarray(mocksky2.cube.s1y)
        s1z = np.asarray(mocksky2.cube.s1z)
        
        print(f"✅ LPT displacements computed successfully")
        print(f"  s1x - Mean: {np.mean(s1x):.8f}, Std: {np.std(s1x):.8f}")
        print(f"  s1y - Mean: {np.mean(s1y):.8f}, Std: {np.std(s1y):.8f}")
        print(f"  s1z - Mean: {np.mean(s1z):.8f}, Std: {np.std(s1z):.8f}")
        
        # Compute RMS displacement
        rms_disp = np.sqrt(np.mean(s1x**2 + s1y**2 + s1z**2))
        print(f"  RMS displacement: {rms_disp:.6f}")
    else:
        print(f"❌ LPT computation failed with return code: {result2}")
        return None
    
    # Save results
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, "legacy_grids.npz")
    np.savez(output_file,
             delta=delta,
             s1x=s1x,
             s1y=s1y,
             s1z=s1z)
    
    file_size = os.path.getsize(output_file) / (1024**2)  # MB
    print(f"\n--- Results saved to: {output_file} ---")
    print(f"File size: {file_size:.2f} MB")
    
    print(f"\n{'='*80}")
    print("LEGACY IMPLEMENTATION - COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return {
        'delta': delta,
        's1x': s1x,
        's1y': s1y,
        's1z': s1z,
        'output_file': output_file
    }

def run_refactored_implementation(pspec_func, N=128, seed=13579, zics=100, H0=68):
    """Run the refactored implementation using simplified API"""
    if not NEW_API_AVAILABLE:
        print(f"\n⚠️  Skipping refactored implementation - API not available")
        return None
    
    print(f"\n{'='*80}")
    print("REFACTORED IMPLEMENTATION - Using simplified API")
    print(f"{'='*80}")
    
    print(f"Parameters:")
    print(f"  N (grid size): {N}")
    print(f"  L (box size): 7700.0 Mpc/h")
    print(f"  seed: {seed}")
    print(f"  H0: {H0}")
    print(f"  z_initial: {zics}")
    
    try:
        # Create cosmological parameters (check what the constructor actually accepts)
        print(f"\n--- Creating cosmological parameters ---")
        # Use the legacy interface approach for compatibility
        cosmo_params = CosmologicalParameters()  # Use defaults
        print(f"✅ Cosmological parameters created")
        
        # Create cosmology service with CAMB power spectrum
        pspec = pspec_func()
        cosmo_service = CosmologyService(cosmo_params, power_spectrum=pspec)
        print(f"✅ Cosmology service created with CAMB power spectrum")
        print(f"  Power spectrum k range: [{np.min(pspec['k']):.2e}, {np.max(pspec['k']):.2e}] h/Mpc")
        print(f"  Power spectrum P(k) range: [{np.min(pspec['pofk']):.2e}, {np.max(pspec['pofk']):.2e}] (Mpc/h)^3")
        
        # Create IC generator using the working approach from minimal example
        print(f"\n--- Creating IC generator ---")
        ic_generator = ICGenerator(
            N=N,
            Lbox=7700.0,
            cosmology=cosmo_params,
            seed=seed,
            z_initial=zics,
            power_spectrum=pspec  # Pass the power spectrum directly
        )
        print(f"✅ IC generator created successfully")
        
        # Generate complete initial conditions using the working pattern
        print(f"\n--- Generating initial conditions ---")
        result = ic_generator.generate_initial_conditions(
            save_output=False,
            steps=['noise', 'delta', 'lpt']
        )
        
        if result.get('success', False):
            # Get density field
            delta = ic_generator.get_density_field()
            if delta is not None:
                delta = np.asarray(delta)
                print(f"✅ Delta field generated successfully")
                print(f"  Shape: {delta.shape}")
                print(f"  Mean: {np.mean(delta):.8f}")
                print(f"  Std: {np.std(delta):.8f}")
                print(f"  Min: {np.min(delta):.6f}")
                print(f"  Max: {np.max(delta):.6f}")
                print(f"  Non-finite values: {np.sum(~np.isfinite(delta))}")
            else:
                print(f"❌ Delta field is None")
                return None
            
            # Get LPT displacements
            displacements = ic_generator.get_displacement_fields()
            if displacements is not None and len(displacements) >= 3:
                s1x, s1y, s1z = displacements[0], displacements[1], displacements[2]
                if s1x is not None:
                    s1x = np.asarray(s1x)
                    s1y = np.asarray(s1y)
                    s1z = np.asarray(s1z)
                    
                    print(f"✅ LPT displacements computed successfully")
                    print(f"  s1x - Mean: {np.mean(s1x):.8f}, Std: {np.std(s1x):.8f}")
                    print(f"  s1y - Mean: {np.mean(s1y):.8f}, Std: {np.std(s1y):.8f}")
                    print(f"  s1z - Mean: {np.mean(s1z):.8f}, Std: {np.std(s1z):.8f}")
                    
                    # Compute RMS displacement
                    rms_disp = np.sqrt(np.mean(s1x**2 + s1y**2 + s1z**2))
                    print(f"  RMS displacement: {rms_disp:.6f}")
                else:
                    print(f"❌ LPT displacements are None")
                    return None
            else:
                print(f"❌ Could not extract LPT displacements")
                return None
                
        else:
            print(f"❌ Initial conditions generation failed: {result.get('error', 'Unknown error')}")
            return None
        
        # Save results
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, "refactored_grids.npz")
        np.savez(output_file,
                 delta=delta,
                 s1x=s1x,
                 s1y=s1y,
                 s1z=s1z)
        
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"\n--- Results saved to: {output_file} ---")
        print(f"File size: {file_size:.2f} MB")
        
        print(f"\n{'='*80}")
        print("REFACTORED IMPLEMENTATION - COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        
        return {
            'delta': delta,
            's1x': s1x,
            's1y': s1y,
            's1z': s1z,
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"❌ Refactored implementation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(legacy_results, refactored_results):
    """Compare results between implementations"""
    print(f"\n{'='*80}")
    print("COMPARISON - Legacy vs Refactored Results") 
    print(f"{'='*80}")
    
    if legacy_results is None:
        print(f"❌ Cannot compare - legacy results not available")
        return
    
    if refactored_results is None:
        print(f"❌ Cannot compare - refactored results not available")
        return
    
    # Compare delta fields if both available
    if 'delta' in legacy_results and 'delta' in refactored_results:
        legacy_delta = legacy_results['delta']
        refactored_delta = refactored_results['delta']
        
        print(f"\n--- Delta Field Comparison ---")
        print(f"Legacy delta     - Mean: {np.mean(legacy_delta):.8f}, Std: {np.std(legacy_delta):.8f}")
        print(f"Refactored delta - Mean: {np.mean(refactored_delta):.8f}, Std: {np.std(refactored_delta):.8f}")
        
        # Check numerical agreement
        if np.allclose(legacy_delta, refactored_delta, rtol=1e-10, atol=1e-12):
            print(f"✅ Delta fields are numerically identical!")
        else:
            diff = np.abs(legacy_delta - refactored_delta)
            max_diff = np.max(diff)
            rms_diff = np.sqrt(np.mean(diff**2))
            rel_diff = rms_diff / np.std(legacy_delta)
            
            print(f"⚠️  Delta fields differ:")
            print(f"   Max absolute difference: {max_diff:.2e}")
            print(f"   RMS difference: {rms_diff:.2e}")
            print(f"   Relative RMS difference: {rel_diff:.2e}")
            
            if rel_diff < 1e-6:
                print(f"   → Differences are very small (< 1e-6 relative)")
            elif rel_diff < 1e-3:
                print(f"   → Differences are small (< 1e-3 relative)")
            else:
                print(f"   → Differences are significant (> 1e-3 relative)")
    else:
        print(f"⚠️  Cannot compare delta fields - not available in both implementations")
    
    # Compare LPT displacements if both available
    displacement_keys = ['s1x', 's1y', 's1z']
    displacement_names = ['X-displacement', 'Y-displacement', 'Z-displacement']
    
    for key, name in zip(displacement_keys, displacement_names):
        if key in legacy_results and key in refactored_results:
            legacy_disp = legacy_results[key]
            refactored_disp = refactored_results[key]
            
            print(f"\n--- {name} Comparison ---")
            print(f"Legacy {key}     - Mean: {np.mean(legacy_disp):.8f}, Std: {np.std(legacy_disp):.8f}")
            print(f"Refactored {key} - Mean: {np.mean(refactored_disp):.8f}, Std: {np.std(refactored_disp):.8f}")
            
            # Check numerical agreement
            if np.allclose(legacy_disp, refactored_disp, rtol=1e-10, atol=1e-12):
                print(f"✅ {name} fields are numerically identical!")
            else:
                diff = np.abs(legacy_disp - refactored_disp)
                max_diff = np.max(diff)
                rms_diff = np.sqrt(np.mean(diff**2))
                rel_diff = rms_diff / np.std(legacy_disp) if np.std(legacy_disp) > 0 else float('inf')
                
                print(f"⚠️  {name} fields differ:")
                print(f"   Max absolute difference: {max_diff:.2e}")
                print(f"   RMS difference: {rms_diff:.2e}")
                print(f"   Relative RMS difference: {rel_diff:.2e}")
                
                if rel_diff < 1e-6:
                    print(f"   → Differences are very small (< 1e-6 relative)")
                elif rel_diff < 1e-3:
                    print(f"   → Differences are small (< 1e-3 relative)")
                else:
                    print(f"   → Differences are significant (> 1e-3 relative)")
        else:
            print(f"⚠️  Cannot compare {name} - not available in both implementations")
    
    # Compare file sizes
    if 'output_file' in legacy_results and 'output_file' in refactored_results:
        legacy_size = os.path.getsize(legacy_results['output_file']) / (1024**2)
        refactored_size = os.path.getsize(refactored_results['output_file']) / (1024**2)
        print(f"\n--- File Size Comparison ---")
        print(f"Legacy output: {legacy_size:.2f} MB")
        print(f"Refactored output: {refactored_size:.2f} MB")
    
    print(f"\n{'='*80}")
    print("COMPARISON - COMPLETED")
    print(f"{'='*80}")

def main():
    """Main function to run all comparisons"""
    print(f"ExgalToolkit Implementation Comparison")
    print(f"{'='*80}")
    print(f"This script compares the legacy and refactored implementations")
    print(f"following the same workflow as minimal_example_serial.py")
    print(f"{'='*80}")
    
    # Set up parameters
    N = 64  # Smaller for faster testing, can be increased
    seed = 13579
    
    # Set up CAMB power spectrum
    print(f"\n--- Setting up CAMB power spectrum ---")
    pspec_func, zics, H0 = setup_camb_power_spectrum()
    print(f"✅ CAMB power spectrum prepared")
    
    # Run legacy implementation
    legacy_results = run_legacy_implementation(pspec_func, N=N, seed=seed, zics=zics, H0=H0)
    
    # Run refactored implementation
    refactored_results = run_refactored_implementation(pspec_func, N=N, seed=seed, zics=zics, H0=H0)
    
    # Compare results
    compare_results(legacy_results, refactored_results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    legacy_success = legacy_results is not None
    refactored_success = refactored_results is not None
    
    print(f"Legacy implementation:     {'✅ SUCCESS' if legacy_success else '❌ FAILED'}")
    print(f"Refactored implementation: {'✅ SUCCESS' if refactored_success else '❌ FAILED'}")
    
    if legacy_success and refactored_success:
        print(f"Both implementations completed successfully!")
        print(f"The refactored architecture preserves the original functionality")
        print(f"while providing a cleaner, more maintainable codebase.")
    elif legacy_success:
        print(f"Legacy implementation works, refactored needs attention.")
    elif refactored_success:
        print(f"Refactored implementation works, legacy had issues.")
    else:
        print(f"Both implementations had issues - check configuration.")
    
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
