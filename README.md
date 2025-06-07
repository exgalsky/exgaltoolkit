# EXtra-GALactic (sky simulation) TOOLKIT

### An assortment of tools and utilities needed for extra-galactic sky simulations with lagrangian perturbation theory

## Overview

This toolkit provides a comprehensive framework for generating mock galaxy catalogs and initial conditions using Lagrangian Perturbation Theory (LPT). The codebase has been recently refactored to provide a modern, maintainable architecture while preserving 100% backward compatibility with existing user code.

## Architecture

The toolkit now features a clean modular architecture with the following components:

### Core Modules
- **`exgaltoolkit.core`** - Configuration management, data models, and exception handling
- **`exgaltoolkit.services`** - Business logic services (cosmology, grid operations, LPT, I/O)
- **`exgaltoolkit.workflow`** - Step-based simulation execution pipeline
- **`exgaltoolkit.api`** - High-level simulation interfaces and factory patterns
- **`exgaltoolkit.legacy`** - Backward compatibility wrappers for existing interfaces

### Legacy Modules (Fully Compatible)
- **`exgaltoolkit.mockgen`** - Mock galaxy generation (Sky, CosmologyInterface, ICs)
- **`exgaltoolkit.lpt`** - Lagrangian Perturbation Theory implementation (Cube)
- **`exgaltoolkit.util`** - Utilities for MPI, JAX, and backend management
- **`exgaltoolkit.mathutil`** - Mathematical utilities and random number generation

## Installation

To install simply do `pip install .` at the top level directory.

```bash
# Standard installation
pip install .

# Development installation (editable)
pip install -e .
```

## Usage

### New Modern API (Recommended for new projects)

```python
from exgaltoolkit.api import SimulationFactory

# Create a simulation with clean configuration
factory = SimulationFactory()
simulation = factory.create_mock_generation_simulation(
    N=128,           # Grid size
    L=7700.0,        # Box size in Mpc/h
    H0=70.0,         # Hubble constant
    Omega_m=0.3,     # Matter density parameter
    seed=12345       # Random seed
)

# Run the simulation
result = simulation.run()

# Access results
grid_data = simulation.get_grid_data()
positions, velocities = simulation.get_particle_data()
```

### Legacy API (Existing code continues to work)

```python
# All existing code works unchanged
from exgaltoolkit.mockgen import Sky, CosmologyInterface
from exgaltoolkit.lpt import Cube
from exgaltoolkit.mockgen import ICs

# Original interface still works
sky = Sky(N=128, L=7700.0, seed=12345)
cube = Cube(sky)
ics = ICs(cube)
# ... etc
```

## Testing

The toolkit includes comprehensive test suites to ensure reliability and compatibility:

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_minimal_example.py
pytest tests/test_refactored_architecture.py

# Run with verbose output to see detailed information
pytest -v -s tests/test_minimal_example.py

# Run comparison tests with detailed output (standalone script)
python tests/run_comparison_tests.py
```

### Test Suites

#### 1. Minimal Example Tests (`test_minimal_example.py`)
Tests the basic functionality and workflow using the legacy interface:
- Serial execution pipeline
- Sky generation and initialization
- Cube creation and LPT computation
- Initial conditions generation
- Output file creation

#### 2. Refactored Architecture Tests (`test_refactored_architecture.py`)
Comprehensive tests for the new modular architecture:

**Configuration System Tests:**
- `test_simulation_config_creation` - Validates configuration object creation
- `test_simulation_factory_from_config` - Tests factory pattern with configurations
- `test_simulation_factory_from_legacy_kwargs` - Tests legacy parameter conversion

**Execution Pipeline Tests:**
- `test_simulation_execution` - End-to-end simulation execution
- `test_legacy_compatibility_maintained` - Ensures legacy interfaces work
- `test_backward_compatibility_wrapper` - Tests wrapper fallback mechanisms

**Service Integration Tests:**
- `test_cosmology_service_integration` - Cosmological parameter handling
- `test_grid_service_integration` - Grid generation and FFT operations

#### 3. Enhanced Minimal Example Tests (`test_minimal_example.py`)
Extended tests that replicate `minimal_example_serial.py` with detailed output:

**Detailed Information Tests:**
- `test_print_helpful_information_legacy` - Shows step-by-step legacy execution with statistics
- `test_print_helpful_information_refactored` - Shows new API execution with detailed output
- `test_compare_legacy_vs_refactored` - Side-by-side comparison of both implementations

**Comparison Features:**
- Detailed parameter display
- Step-by-step execution logging
- Statistical analysis of generated fields
- File output verification
- Numerical comparison between implementations

#### 4. Standalone Comparison Script (`run_comparison_tests.py`)
Comprehensive standalone script for detailed implementation comparison:

```bash
# Run detailed comparison with full output
python tests/run_comparison_tests.py
```

**Features:**
- CAMB power spectrum setup exactly like `minimal_example_serial.py`
- Side-by-side execution of legacy and refactored implementations
- Detailed statistical analysis and comparison
- File size and performance metrics
- Numerical accuracy verification

### Test Results

Latest test run results:
```
============================= test session starts ==============================
tests/test_minimal_example.py .......                                    [ 46%]
tests/test_refactored_architecture.py ........                           [100%]

============================= 15 passed in 42.79s ==============================
```

**15/15 tests passing** - Full validation of both legacy and modern interfaces.

## Development

### Project Structure

```
exgaltoolkit/
├── core/                    # Configuration and data models
├── services/               # Business logic services  
├── workflow/               # Execution pipeline
├── api/                    # High-level interfaces
├── legacy/                 # Backward compatibility
├── mockgen/               # Legacy mock generation
├── lpt/                   # Legacy LPT implementation
├── util/                  # Utilities and backends
└── mathutil/              # Mathematical operations
```

### Key Features

**Separation of Concerns**: Clean modular architecture with distinct responsibilities
**Backward Compatibility**: 100% compatibility with existing user code
**Modern Patterns**: Factory patterns, dependency injection, workflow pipelines
**Comprehensive Testing**: Full test coverage for all interfaces and functionality
**Type Safety**: Complete type annotations throughout the codebase
**Documentation**: Comprehensive docstrings and error messages

### Debugging and Troubleshooting

To test the namespace of the installed package:
```bash
python ./tests/traverse_namespace.py
```

For development debugging:
```bash
# Enable verbose logging
export EXGAL_LOG_LEVEL=DEBUG

# Run tests with detailed output
pytest -v -s tests/
```

## Migration Guide

**For existing users**: No changes required - all existing code continues to work unchanged.

**For new development**: Consider using the modern API for better error handling, validation, and maintainability.

See `refactor-summary-1.md` for detailed information about the architectural changes and migration options.