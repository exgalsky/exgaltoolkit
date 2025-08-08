# JAX-ICs: Agentic Development Specification
## A Comprehensive Framework for Agent-Driven Development of a JAX-based Cosmological Initial Conditions Generator

### 1. Executive Summary

This specification provides a complete framework for developing JAX-ICs using agentic development tools. It serves as the single source of truth for AI agents to develop, validate, and maintain a distributed, JAX-accelerated cosmological initial conditions generator. The document emphasizes agent-verifiable requirements, explicit validation criteria, and clear architectural boundaries to enable autonomous development with minimal human intervention.

### 2. Mission Statement

Develop a production-ready, platform-independent cosmological initial conditions generator that:
- Leverages JAX for automatic differentiation and GPU/TPU acceleration
- Generates reproducible initial conditions for N-body simulations
- Scales efficiently across multiple GPUs with identical results
- Provides a simple, well-documented Python API
- Ensures numerical precision suitable for cosmological simulations

### 3. Core Products and Deliverables

#### 3.1 Primary Components
- **Core Library** (`jax_ics/`): JAX-based implementation of IC generation algorithms
- **API Interface** (`jax_ics/api/`): High-level Python API for users
- **Validation Suite** (`jax_ics/validation/`): Automated testing framework
- **Example Scripts** (`examples/`): Ready-to-run demonstration code
- **Documentation** (`docs/`): Comprehensive user and developer documentation

#### 3.2 Key Classes and Methods

```python
# Primary API classes
class CosmologyParams:
    """Container for cosmological parameters"""
    omega_m: float
    omega_lambda: float
    h: float
    n_s: float
    sigma_8: float

class ICGenerator:
    """Main interface for generating initial conditions"""
    def __init__(self, cosmology: CosmologyParams)
    def generate_density_field(self, box_size, grid_resolution, seed) -> jax.Array
    def generate_lpt_positions(self, density_field, redshift) -> jax.Array
    def validate_consistency(self) -> ValidationReport

class PowerSpectrum:
    """Power spectrum computation and manipulation"""
    def compute_linear_power(self, k_array) -> jax.Array
    def apply_transfer_function(self, k_array) -> jax.Array
```

### 4. Functional Requirements

#### 4.1 Core Functionality

**FR-001: Gaussian Random Field Generation**
- Generate 3D Gaussian random density fields on uniform grids
- Support grid resolutions from 64³ to 4096³
- Ensure hermitian symmetry in Fourier space
- Validation: Compare power spectrum of generated field to input

**FR-002: Lagrangian Perturbation Theory (LPT)**
- Implement first-order LPT (Zel'dovich approximation)
- Implement second-order LPT corrections
- Support both particle and grid-based outputs
- Validation: Verify momentum conservation to machine precision

**FR-003: Platform Independence**
- Identical results for same (power_spectrum, seed, box_size, resolution)
- Support CPU, single GPU, and multi-GPU execution
- Use JAX's PRNG for reproducible random numbers
- Validation: Bit-wise comparison across platforms

**FR-004: Power Spectrum Support**
- Linear power spectrum from transfer functions (Eisenstein & Hu)
- Non-linear corrections (Halofit)
- User-provided tabulated power spectra
- Validation: Accuracy within 1% of reference implementations

#### 4.2 Performance Requirements

**PR-001: Scalability**
- Strong scaling efficiency >80% up to 32 GPUs
- Weak scaling efficiency >90% up to 128 GPUs
- Memory usage <1.5x theoretical minimum

**PR-002: Speed Benchmarks**
- 512³ density field generation: <1 second on single V100
- 1024³ LPT displacement: <10 seconds on single V100
- Linear speedup with number of GPUs for large problems

### 5. Software Architecture

#### 5.1 High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                  │
│                  (Python API, CLI tools)                 │
├─────────────────────────────────────────────────────────┤
│                    Core Algorithms Layer                 │
│         (Field Generation, LPT, Power Spectra)          │
├─────────────────────────────────────────────────────────┤
│                   JAX Acceleration Layer                 │
│           (JIT compilation, vmap, pmap, sharding)       │
├─────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                    │
│        (Memory management, I/O, MPI communication)       │
└─────────────────────────────────────────────────────────┘
```

#### 5.2 Core Algorithm Pseudocode

```python
# Density Field Generation
def generate_density_field(power_spectrum, box_size, grid_resolution, seed):
    # 1. Initialize k-space grid
    k_grid = create_k_space_grid(box_size, grid_resolution)
    
    # 2. Generate complex Gaussian random field
    key1, key2 = jax.random.split(seed)
    real_part = jax.random.normal(key1, shape=k_grid.shape)
    imag_part = jax.random.normal(key2, shape=k_grid.shape)
    delta_k = (real_part + 1j * imag_part) / sqrt(2)
    
    # 3. Apply power spectrum
    delta_k *= sqrt(power_spectrum(k_grid))
    
    # 4. Enforce Hermitian symmetry
    delta_k = enforce_hermitian_symmetry(delta_k)
    
    # 5. Transform to real space
    delta_x = jax.numpy.fft.ifftn(delta_k).real
    
    return delta_x

# LPT Displacement Calculation
def compute_lpt_displacement(density_field, growth_factor):
    # 1. Compute gravitational potential
    phi_k = -density_field_k / k_squared
    
    # 2. Compute displacement via gradient
    displacement = []
    for axis in [0, 1, 2]:
        grad_phi_k = 1j * k[axis] * phi_k
        displacement.append(growth_factor * ifftn(grad_phi_k).real)
    
    return jax.numpy.stack(displacement, axis=-1)
```

#### 5.3 Multi-GPU Distribution Strategy

```python
# Data parallelism using JAX pmap
@jax.pmap
def distributed_density_generation(local_seed, global_params):
    # Each device generates its local portion
    local_field = generate_local_density_field(
        local_seed, global_params
    )
    return local_field

# Domain decomposition for large grids
def domain_decompose(global_grid_shape, num_devices):
    # Slab decomposition along z-axis
    local_shape = (
        global_grid_shape[0],
        global_grid_shape[1],
        global_grid_shape[2] // num_devices
    )
    return local_shape
```

### 6. Agent-Specific Development Guidelines

#### 6.1 Code Organization Standards

**Directory Structure:**
```
jax-ics/
├── jax_ics/
│   ├── __init__.py
│   ├── core/
│   │   ├── density_field.py    # Gaussian field generation
│   │   ├── lpt.py             # LPT implementations
│   │   └── power_spectrum.py   # Power spectrum utilities
│   ├── distributed/
│   │   ├── communication.py    # MPI/NCCL wrappers
│   │   └── decomposition.py    # Domain decomposition
│   ├── api/
│   │   └── generator.py        # High-level API
│   └── validation/
│       ├── tests.py           # Unit tests
│       └── benchmarks.py      # Performance tests
├── examples/
│   ├── basic_usage.py
│   ├── multi_gpu_example.py
│   └── validation_suite.py
├── tests/
│   ├── test_density_field.py
│   ├── test_lpt.py
│   └── test_scaling.py
└── docs/
    ├── user_guide.md
    ├── api_reference.md
    └── developer_guide.md
```

#### 6.2 Agent Interaction Patterns

**Code Generation Guidelines:**
- Always include type hints for all functions
- Use JAX-compatible operations only (no NumPy in computational code)
- Implement @jax.jit decorators for performance-critical functions
- Add comprehensive docstrings with examples

**Validation First Development:**
1. Write validation test before implementing feature
2. Implement minimal code to pass validation
3. Optimize only after correctness is verified
4. Document performance characteristics

### 7. Explicit Validation Tests

#### 7.1 Correctness Validations

**VAL-001: Power Spectrum Recovery**
```python
def test_power_spectrum_recovery():
    """Verify generated field matches input power spectrum"""
    # Generate field
    field = generator.generate_density_field(...)
    
    # Compute power spectrum
    measured_ps = compute_power_spectrum(field)
    expected_ps = input_power_spectrum(k_values)
    
    # Assert within 5% for k < k_nyquist/2
    assert np.allclose(measured_ps, expected_ps, rtol=0.05)
```

**VAL-002: Multi-GPU Consistency**
```python
def test_multi_gpu_consistency():
    """Ensure identical results across different GPU configurations"""
    # Run on 1, 2, 4, 8 GPUs
    results = {}
    for n_gpus in [1, 2, 4, 8]:
        results[n_gpus] = run_with_gpus(n_gpus, params)
    
    # Compare all results
    reference = results[1]
    for n_gpus, result in results.items():
        assert arrays_equal(result, reference, tolerance=1e-12)
```

**VAL-003: LPT Momentum Conservation**
```python
def test_momentum_conservation():
    """Verify total momentum is zero"""
    positions = generator.generate_lpt_positions(...)
    velocities = compute_velocities(positions)
    total_momentum = np.sum(velocities, axis=0)
    
    assert np.allclose(total_momentum, 0.0, atol=1e-10)
```

#### 7.2 Performance Validations

**PERF-001: Strong Scaling Test**
```python
def test_strong_scaling():
    """Measure speedup with fixed problem size"""
    problem_size = 1024**3
    baseline_time = time_on_gpus(1, problem_size)
    
    for n_gpus in [2, 4, 8, 16, 32]:
        time_n = time_on_gpus(n_gpus, problem_size)
        efficiency = baseline_time / (time_n * n_gpus)
        assert efficiency > 0.8  # 80% efficiency requirement
```

**PERF-002: Memory Usage**
```python
def test_memory_efficiency():
    """Verify memory usage is within bounds"""
    theoretical_memory = calculate_theoretical_memory(params)
    actual_memory = measure_peak_memory(params)
    
    assert actual_memory < 1.5 * theoretical_memory
```

### 8. Agent Collaboration Protocols

#### 8.1 Task Decomposition

**Primary Agent Roles:**

1. **Architecture Agent**: Designs overall system structure
2. **Implementation Agent**: Writes core algorithms
3. **Validation Agent**: Creates and runs tests
4. **Documentation Agent**: Maintains docs and examples
5. **Review Agent**: Checks code quality and standards

#### 8.2 Inter-Agent Communication

```yaml
# Agent task handoff example
task_handoff:
  from: Architecture Agent
  to: Implementation Agent
  task: "Implement distributed FFT wrapper"
  context:
    - Input/output specifications
    - Performance requirements
    - Existing code interfaces
  validation_criteria:
    - Unit tests pass
    - Scales to 32 GPUs
    - Maintains numerical precision
```

### 9. Continuous Validation Framework

#### 9.1 Automated Test Suite

```python
# Run on every code change
validation_pipeline = [
    # Level 1: Unit tests (< 1 minute)
    "pytest tests/unit/",
    
    # Level 2: Integration tests (< 10 minutes)
    "pytest tests/integration/",
    
    # Level 3: Performance tests (< 30 minutes)
    "python -m jax_ics.validation.benchmarks",
    
    # Level 4: Full validation suite (< 2 hours)
    "python -m jax_ics.validation.full_suite"
]
```

#### 9.2 Regression Detection

```python
# Automated performance regression detection
def check_performance_regression(new_results, baseline):
    """Compare against baseline performance"""
    for test_name, new_time in new_results.items():
        baseline_time = baseline[test_name]
        if new_time > 1.1 * baseline_time:  # 10% regression threshold
            raise RegressionError(f"{test_name} regressed by "
                                f"{(new_time/baseline_time - 1)*100:.1f}%")
```

### 10. Documentation Requirements

#### 10.1 Code Documentation Standards

**Function Documentation Template:**
```python
def generate_density_field(
    power_spectrum: Callable[[jnp.ndarray], jnp.ndarray],
    box_size: float,
    grid_resolution: int,
    seed: jax.random.PRNGKey
) -> jnp.ndarray:
    """Generate a Gaussian random density field.
    
    Creates a 3D density field with the specified power spectrum
    using Gaussian random field theory. The field is generated
    in Fourier space and transformed to real space.
    
    Args:
        power_spectrum: Function P(k) returning power at wavenumber k
        box_size: Physical size of the box in Mpc/h
        grid_resolution: Number of grid points per dimension
        seed: JAX PRNG key for reproducible randomness
        
    Returns:
        3D array of shape (grid_resolution,)³ containing density
        contrast δ = (ρ - ρ̄) / ρ̄
        
    Example:
        >>> ps = lambda k: 1000 * (k / 0.1)**-3  # Simple power law
        >>> field = generate_density_field(ps, 100.0, 256, key)
        >>> assert field.shape == (256, 256, 256)
        
    Note:
        The returned field satisfies <δ> = 0 and has the input
        power spectrum to within cosmic variance.
    """
```

#### 10.2 User Documentation Structure

1. **Quick Start Guide**: 5-minute introduction
2. **Installation Guide**: Platform-specific instructions
3. **User Manual**: Complete feature documentation
4. **API Reference**: Auto-generated from docstrings
5. **Theory Guide**: Mathematical background
6. **Performance Guide**: Optimization tips

### 11. Success Metrics

#### 11.1 Development Metrics
- Code coverage > 95%
- Documentation coverage > 90%
- API stability (no breaking changes after v1.0)
- Performance regression < 5% per release

#### 11.2 Scientific Validation
- Agreement with MUSIC/2LPTic codes < 0.1% 
- Published validation paper
- Adoption by >3 major simulation projects
- Community feedback incorporation

### 12. Appendices

#### A. Glossary of Terms
- **LPT**: Lagrangian Perturbation Theory
- **IC**: Initial Conditions  
- **FFT**: Fast Fourier Transform
- **PRNG**: Pseudo-Random Number Generator

#### B. Reference Implementations
- MUSIC: Multi-scale Initial Conditions
- 2LPTic: Second-order LPT code
- monofonIC: High-order LPT implementation

#### C. Mathematical Background
- Gaussian random field theory
- Zel'dovich approximation
- Second-order LPT corrections
- Power spectrum definitions