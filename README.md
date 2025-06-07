# EXtra-GALactic Toolkit

A toolkit for extra-galactic sky simulations with Lagrangian Perturbation Theory

## Overview

This toolkit provides a framework for generating cosmological initial conditions using Lagrangian Perturbation Theory (LPT). The codebase implements core functionality for cosmological simulations with JAX acceleration and MPI support.

## Architecture

The toolkit features a modular architecture with the following components:

### Core Modules
- **`exgaltoolkit.initial_conditions`** - IC generation classes (ICGenerator, ICWriter)
- **`exgaltoolkit.cosmology`** - Cosmological parameters and services (CosmologyService, CosmologicalParameters)
- **`exgaltoolkit.grid`** - Grid operations and LPT calculations (GridOperations, LPTCalculator)
- **`exgaltoolkit.util`** - Utilities for MPI, JAX, and backend management
- **`exgaltoolkit.mathutil`** - Mathematical utilities and random number generation
- **`exgaltoolkit.data`** - Data files and resources

## Installation

To install simply do `pip install .` at the top level directory.

```bash
# Standard installation
pip install .

# Development installation (editable)
pip install -e .
```

## Usage

### Modern API

```python
from exgaltoolkit.initial_conditions import ICGenerator, ICWriter
from exgaltoolkit.cosmology import CosmologyService, CosmologicalParameters
from exgaltoolkit.grid import GridOperations, LPTCalculator

# Create cosmological parameters
cosmo_params = CosmologicalParameters(
    H0=70.0,           # Hubble constant
    Omega_m=0.3,       # Matter density parameter
    Omega_b=0.05,      # Baryon density parameter
    n_s=0.96,          # Scalar spectral index
    sigma_8=0.8        # Amplitude of matter fluctuations
)

# Initialize cosmology service
cosmology = CosmologyService(cosmo_params)

# Create grid operations
grid_ops = GridOperations(
    N=64,              # Grid size
    L=7700.0,          # Box size in Mpc/h
    cosmology=cosmology
)

# Generate initial conditions
ic_generator = ICGenerator(
    grid_ops=grid_ops,
    seed=12345         # Random seed
)

# Run the generation
density_field, velocity_field = ic_generator.generate()

# Write output
ic_writer = ICWriter(grid_ops)
ic_writer.write_fields("output.h5", density_field, velocity_field)
```

## Testing

The toolkit includes comprehensive unit tests to ensure correctness:

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output to see detailed information
pytest -v -s tests/test_initial_conditions.py
```

### Test Suite

#### Initial Conditions Tests (`test_initial_conditions.py`)
Comprehensive tests for the core functionality:

**Basic Functionality Tests:**
- `test_cosmological_parameters_creation` - Validates cosmological parameter setup
- `test_cosmology_service_initialization` - Tests cosmology service creation
- `test_grid_operations_creation` - Validates grid operations setup
- `test_ic_generator_initialization` - Tests IC generator creation

**Integration Tests:**
- `test_density_field_generation` - End-to-end density field generation
- `test_velocity_field_generation` - End-to-end velocity field generation
- `test_full_ic_generation_pipeline` - Complete initial conditions pipeline

**Validation Tests:**
- Density field statistics validation (mean, variance)
- Power spectrum consistency checks
- Output file generation and verification

### Test Parameters

Tests use realistic simulation parameters to ensure proper validation:
- Grid size: N=64 (to minimize finite box effects)
- Box size: L=7700.0 Mpc/h
- CAMB power spectrum data
- Fixed random seed for reproducibility

### Test Results

Latest test validation:
- All density field means within tolerance (< 1e-6)
- Power spectrum consistency verified
- Complete pipeline functionality confirmed

## Development

### Project Structure

```
exgaltoolkit/
├── initial_conditions/        # Main IC generation classes
├── cosmology/                 # Cosmological parameters and services
├── grid/                      # Grid operations and LPT calculations
├── util/                      # MPI, JAX, and backend utilities
├── mathutil/                  # Mathematical operations and RNG
└── data/                      # Data files and resources
```

### Key Features

**Clean Architecture**: Minimal, focused modules with clear responsibilities
**Modern Implementation**: JAX-accelerated computations with MPI support
**Comprehensive Testing**: Full test coverage for core functionality
**Type Safety**: Complete type annotations throughout the codebase
**Documentation**: Comprehensive docstrings and examples

### Examples

See the `examples/` directory for usage examples:
- `minimal_example_serial.py` - Basic serial IC generation example

### Debugging and Troubleshooting

For development debugging:
```bash
# Enable verbose logging
export EXGAL_LOG_LEVEL=DEBUG

# Run tests with detailed output
pytest -v -s tests/
```