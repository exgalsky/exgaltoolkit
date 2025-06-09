# EXtra-GALactic Toolkit

A toolkit for generating cosmological initial conditions using Lagrangian Perturbation Theory

## Overview

ExGalToolkit provides functionality for generating cosmological initial conditions for N-body simulations. The toolkit implements Lagrangian Perturbation Theory calculations using JAX for computational efficiency and supports both analytical and CAMB-generated power spectra.

**Key Features:**
- JAX-based implementation with GPU acceleration support
- Modular architecture with clean API separation
- Numerical compatibility with existing implementations
- CAMB power spectrum integration
- Reproducible results with deterministic random number generation
- Support for first and second-order LPT calculations

## Architecture

The toolkit provides a simplified API built around three core classes:

### Core API Classes
- **`ICGenerator`** - Main class for initial conditions generation
- **`CosmologicalParameters`** - Container for cosmological parameters (Ωₘ, H₀, σ₈, etc.)  
- **`CosmologyService`** - Handles power spectra and cosmological calculations

### Internal Modules
- **`exgaltoolkit.core`** - Core algorithms (FFTs, transfer functions, LPT calculations)
- **`exgaltoolkit.cosmology`** - Cosmological parameter handling and growth factors
- **`exgaltoolkit.grid`** - Grid operations and LPT displacement calculations
- **`exgaltoolkit.util`** - Utilities for JAX backend configuration, timing, and logging

### Implementation Details
The toolkit has been refactored to use JAX for all numerical computations:
- All arrays and operations use JAX for potential GPU acceleration
- Maintains numerical compatibility with previous NumPy-based implementations
- Modular design allows for easy extension and modification of individual components

## Installation

To install simply do `pip install .` at the top level directory.

```bash
# Standard installation
pip install .

# Development installation (editable)
pip install -e .
```

## Usage

### Basic Example

```python
from exgaltoolkit import ICGenerator, CosmologicalParameters

# Create cosmological parameters
cosmo_params = CosmologicalParameters(
    Omega_m=0.31,      # Matter density parameter
    h=0.68,            # Reduced Hubble constant  
    Omega_b=0.049,     # Baryon density parameter
    sigma_8=0.8,       # Amplitude of matter fluctuations
    n_s=0.96           # Scalar spectral index
)

# Generate initial conditions
ic_generator = ICGenerator(
    N=64,              # Grid resolution
    Lbox=7700.0,       # Box size in Mpc
    cosmology=cosmo_params,
    seed=12345,        # Random seed for reproducibility
    z_initial=100      # Initial redshift
)

# Run the generation
result = ic_generator.generate_initial_conditions()

# Access results
density_field = ic_generator.get_density_field()
displacements = ic_generator.get_displacement_fields()  # LPT displacements

print(f"Generated density field with shape: {density_field.shape}")
print(f"Mean density contrast: {np.mean(density_field):.2e}")
print(f"RMS density contrast: {np.std(density_field):.3f}")
```

### CAMB Integration

```python
import camb
from exgaltoolkit import ICGenerator, CosmologicalParameters, CosmologyService

# Set up CAMB power spectrum
camb_pars = camb.set_params(H0=68, ombh2=0.022, omch2=0.12)
camb_pars.set_matter_power(redshifts=[100], kmax=2.0)
camb_results = camb.get_results(camb_pars)

# Extract power spectrum
k, z, pk = camb_results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e2, npoints=2000)
power_spectrum = {'k': k, 'pofk': pk[0, :]}

# Create cosmology service with CAMB power spectrum
cosmo_params = CosmologicalParameters(Omega_m=0.31, h=0.68)
cosmology_service = CosmologyService(cosmo_params, power_spectrum=power_spectrum)

# Generate initial conditions
ic_generator = ICGenerator(
    N=128, 
    Lbox=7700.0, 
    cosmology=cosmo_params,
    seed=12345,
    lpt_order=2        # Use 2nd-order LPT
)

# Override with CAMB cosmology
ic_generator.cosmology_service = cosmology_service

# Generate with file output
result = ic_generator.generate_initial_conditions(
    save_output=True,
    output_filename="my_ics"
)
```

## Testing

The toolkit includes unit tests that validate the JAX implementation against established results:

### Running Tests

```bash
# Run all tests  
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_core_functionality.py::TestBasicAPI -v

# Run with detailed output and timing
python -m pytest tests/ -v -s
```

### Test Suite

#### Core Functionality Tests (`test_core_functionality.py`)

The test suite validates the refactored JAX implementation:

**Component Tests (`TestBasicAPI`):**
- `test_api_components_creation` - API component instantiation and configuration
- `test_basic_ic_generation` - Basic density field generation with simple power spectra
- `test_displacement_generation` - LPT displacement field computation (1st and 2nd order)
- `test_reproducibility` - Deterministic results with fixed random seeds

**CAMB Integration Tests (`TestCAMBIntegration`):**
- `test_camb_power_spectrum_integration` - CAMB power spectrum handling
- `test_full_camb_workflow` - End-to-end workflow with CAMB power spectra
- `test_camb_density_field_statistics` - Statistical validation of CAMB-generated fields

**Numerical Accuracy Tests:**
- `test_numerical_accuracy_debug` - Precision validation against target values
- `test_jax_vs_legacy_numerical_accuracy` - Component-by-component accuracy verification

### Test Configuration

Tests use a centralized configuration system with realistic parameters:

```python
TEST_CONFIG = {
    'N': 64,                      # Grid resolution
    'Lbox': 7700.0,              # Box size (Mpc)
    'seed': 13579,               # Fixed random seed
    'z_initial': 100,            # Initial redshift
    'H0': 68,                    # Hubble constant
    'Omega_m': 0.31,             # Matter density
    'target_mean': -1.64e-08,    # Expected density field mean
    'target_std': 1.40e-03,      # Expected density field RMS
}
```

### Validation Results

**Numerical Accuracy**: All tests pass with the refactored JAX implementation
- Density field statistics match targets within tolerance
- Numerical compatibility with original implementation verified
- GPU acceleration functional on CUDA devices

**Performance**: Timing benchmarks on GPU hardware
- Noise generation: ~0.8s 
- Density computation: ~0.9s
- LPT computation: ~0.6s  
- Particle computation: ~1.3s

**Reproducibility**: Deterministic results across multiple runs
- Fixed seeds produce identical outputs
- Cross-platform consistency verified

## Development

### Project Structure

```
exgaltoolkit/
├── core/                      # Core JAX-based algorithms
│   ├── transfers.py          #   Transfer functions and power spectrum application
│   ├── noise.py              #   White noise generation
│   ├── fft_ops.py            #   FFT operations and k-space utilities  
│   ├── lpt_math.py           #   LPT displacement calculations
│   └── k_grids.py            #   k-space grid generation
├── cosmology/                 # Cosmological parameter handling
│   ├── parameters.py         #   CosmologicalParameters class
│   ├── modern_service.py     #   CosmologyService class
│   └── growth.py             #   Growth factor calculations
├── initial_conditions/        # Main IC generation API
│   ├── generator.py          #   ICGenerator class
│   └── writer.py             #   Output file writing
├── grid/                      # Grid operations and LPT
│   ├── operations.py         #   Grid manipulation utilities
│   └── lpt.py                #   LPT displacement computation
├── util/                      # Utilities and backend management
│   ├── log_util.py           #   Timing and logging utilities
│   ├── backend.py            #   JAX backend configuration
│   └── jax_util.py           #   JAX-specific utilities
└── data/                      # Static data files
    └── camb_*.dat            #   CAMB power spectrum data
```

### Implementation Notes

**JAX Migration**: 
- All numerical computations use JAX arrays and functions
- GPU acceleration available when supported hardware is detected
- Maintains numerical compatibility with previous NumPy-based implementations

**Code Organization**:
- Cleaned up function and constant names for consistency
- Removed temporary debugging and comparison code from the refactoring process
- Streamlined API with focused class responsibilities
- Centralized configuration management for testing

**Testing Infrastructure**:
- Comprehensive test suite with centralized configuration
- Numerical accuracy validation against established target values
- Performance benchmarking and cross-platform reproducibility testing

### GPU Acceleration

The toolkit automatically detects and utilizes available GPU resources:

```python
# Check available devices
import jax
print("Available devices:", jax.devices())

# Force GPU usage (optional)
import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
```

### Debugging and Troubleshooting

For development and debugging:
```bash
# Enable verbose logging  
export EXGAL_LOG_LEVEL=DEBUG

# Run tests with detailed output
python -m pytest tests/ -v -s

# Check GPU utilization (NVIDIA)
nvidia-smi
```

### Performance Considerations

The JAX implementation includes several optimizations:
- JIT compilation for computational kernels
- Efficient memory usage patterns
- Optimized FFT operations using JAX's built-in functions
- Appropriate GPU memory management for large simulations