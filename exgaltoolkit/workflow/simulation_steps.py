"""
Individual simulation step implementations.
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
import numpy as np
from ..core.data_models import SimulationContext, StepResult, ParticleData
from ..core.exceptions import SimulationError
from ..services import (
    GridManager, 
    NoiseGenerator, 
    FFTProcessor, 
    LPTCalculator,
    CosmologyService,
    PowerSpectrumService,
    OutputManager
)

class SimulationStep(ABC):
    """Abstract base class for simulation steps."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this step."""
        pass
    
    @abstractmethod
    def execute(self, context: SimulationContext) -> StepResult:
        """Execute this step with the given context."""
        pass
    
    @abstractmethod
    def validate_prerequisites(self, context: SimulationContext) -> bool:
        """Check if prerequisites for this step are met."""
        pass

class NoiseGenerationStep(SimulationStep):
    """Generate white noise field."""
    
    @property
    def name(self) -> str:
        return "noise"
    
    def validate_prerequisites(self, context: SimulationContext) -> bool:
        """Check if prerequisites for this step are met."""
        return True  # No prerequisites for noise generation
    
    def execute(self, context: SimulationContext) -> StepResult:
        """Execute noise generation step."""
        try:
            config = context.config
            
            # Initialize grid manager if not already done
            if not context.has('grid_manager'):
                grid_manager = GridManager(config.grid)
                context.set('grid_manager', grid_manager)
            else:
                grid_manager = context.get('grid_manager')
            
            # Initialize noise generator
            noise_gen = NoiseGenerator(grid_manager)
            
            # Generate noise for each iteration
            for i in range(config.simulation.n_iterations):
                seed = config.simulation.seed + i
                noise_field = noise_gen.generate_white_noise(seed)
                context.set(f'noise_field_{i}', noise_field)
            
            # For backward compatibility, store the last noise field as 'delta'
            if config.simulation.n_iterations > 0:
                context.set('delta', context.get(f'noise_field_{config.simulation.n_iterations-1}'))
            
            return StepResult(
                success=True,
                step_name=self.name,
                message=f"Generated noise for {config.simulation.n_iterations} iterations"
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                message=f"Noise generation failed: {str(e)}"
            )

class ConvolutionStep(SimulationStep):
    """Apply power spectrum convolution to get delta field."""
    
    @property
    def name(self) -> str:
        return "convolution"
    
    def validate_prerequisites(self, context: SimulationContext) -> bool:
        """Check if prerequisites for this step are met."""
        return context.has('delta')
    
    def execute(self, context: SimulationContext) -> StepResult:
        """Execute convolution step."""
        try:
            config = context.config
            grid_manager = context.get('grid_manager')
            
            # Initialize services
            if config.power_spectrum:
                power_service = PowerSpectrumService(config.power_spectrum)
            else:
                power_service = PowerSpectrumService()  # Uses default
            
            fft_processor = FFTProcessor(grid_manager)
            
            # Process each noise field
            for i in range(config.simulation.n_iterations):
                noise_key = f'noise_field_{i}'
                if context.has(noise_key):
                    noise_field = context.get(noise_key)
                    
                    # Get transfer function
                    transfer_data = power_service.get_transfer_function(
                        None, grid_manager.d3k, grid_manager.N
                    )
                    
                    # Apply convolution: noise -> k-space -> apply transfer -> real space
                    noise_k = fft_processor.forward_transform(noise_field)
                    convolved_k = fft_processor.apply_transfer_function(noise_k, transfer_data)
                    delta_field = fft_processor.inverse_transform(convolved_k)
                    
                    context.set(f'delta_field_{i}', delta_field)
            
            # For backward compatibility, store the last delta field as 'delta'
            if config.simulation.n_iterations > 0:
                context.set('delta', context.get(f'delta_field_{config.simulation.n_iterations-1}'))
            
            return StepResult(
                success=True,
                step_name=self.name,
                message="Convolution completed successfully"
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                message=f"Convolution failed: {str(e)}"
            )

class LPTDisplacementStep(SimulationStep):
    """Calculate LPT displacement fields."""
    
    @property
    def name(self) -> str:
        return "LPT"
    
    def validate_prerequisites(self, context: SimulationContext) -> bool:
        """Check if prerequisites for this step are met."""
        return context.has('delta')
    
    def execute(self, context: SimulationContext) -> StepResult:
        """Execute LPT displacement calculation."""
        try:
            config = context.config
            grid_manager = context.get('grid_manager')
            
            # Skip if LPT order is 0 or negative
            if config.simulation.lpt_order <= 0:
                return StepResult(
                    success=True,
                    step_name=self.name,
                    message="LPT order <= 0, skipping displacement calculation"
                )
            
            # Initialize LPT calculator
            lpt_calc = LPTCalculator(grid_manager, config.simulation.lpt_order)
            
            # Process each delta field
            for i in range(config.simulation.n_iterations):
                delta_key = f'delta_field_{i}'
                if context.has(delta_key):
                    delta_field = context.get(delta_key)
                    
                    # Compute LPT displacements
                    input_mode = 'delta' if context.has('delta_field_0') else config.simulation.input_type
                    displacements = lpt_calc.compute_displacements(delta_field, input_mode)
                    
                    context.set(f'lpt_displacements_{i}', displacements)
            
            # For backward compatibility, store the last displacement fields
            if config.simulation.n_iterations > 0:
                last_disp = context.get(f'lpt_displacements_{config.simulation.n_iterations-1}')
                context.set('s1x', last_disp.s1x)
                context.set('s1y', last_disp.s1y)
                context.set('s1z', last_disp.s1z)
                if last_disp.s2x is not None:
                    context.set('s2x', last_disp.s2x)
                    context.set('s2y', last_disp.s2y)
                    context.set('s2z', last_disp.s2z)
            
            return StepResult(
                success=True,
                step_name=self.name,
                message="LPT displacement calculation completed"
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                message=f"LPT calculation failed: {str(e)}"
            )

class InitialConditionsWriteStep(SimulationStep):
    """Write initial conditions to file."""
    
    @property 
    def name(self) -> str:
        return "writeics"
    
    def validate_prerequisites(self, context: SimulationContext) -> bool:
        """Check if prerequisites for this step are met."""
        config = context.config
        return (config.simulation.write_ics and 
                context.has('s1x') and 
                context.has('s1y') and 
                context.has('s1z'))
    
    def execute(self, context: SimulationContext) -> StepResult:
        """Execute initial conditions writing."""
        try:
            config = context.config
            
            if not config.simulation.write_ics:
                return StepResult(
                    success=True,
                    step_name=self.name,
                    message="Initial conditions writing disabled"
                )
            
            # Initialize cosmology service to get growth factors
            cosmo_service = CosmologyService(config.cosmology)
            growth_factors = cosmo_service.compute_growth_factors()
            
            # Get displacement fields
            s1x = context.get('s1x')
            s1y = context.get('s1y') 
            s1z = context.get('s1z')
            s2x = context.get('s2x')
            s2y = context.get('s2y')
            s2z = context.get('s2z')
            
            # Create particle data using the original ICs logic
            particles = self._create_particle_data(
                config, growth_factors, s1x, s1y, s1z, s2x, s2y, s2z
            )
            
            # Write output
            output_manager = OutputManager(config.output)
            
            # Generate filename similar to original
            filename = f"{config.simulation.model_id}_{config.simulation.seed}_Lbox-{config.grid.Lbox}_N-{config.grid.N}_proc-0"
            
            output_manager.write_initial_conditions(particles, filename)
            
            return StepResult(
                success=True,
                step_name=self.name,
                message=f"Initial conditions written to {filename}"
            )
            
        except Exception as e:
            return StepResult(
                success=False,
                step_name=self.name,
                message=f"Initial conditions writing failed: {str(e)}"
            )
    
    def _create_particle_data(self, config, growth_factors, s1x, s1y, s1z, s2x, s2y, s2z) -> ParticleData:
        """Create particle data from displacement fields (preserves original logic)."""
        # This method preserves the exact logic from the original writeics method
        
        h = config.cosmology.h
        omega_m = config.cosmology.omega_m
        N = config.grid.N
        Lbox = config.grid.Lbox
        z = config.simulation.initial_redshift
        a = 1 / (1 + z)
        
        # Cosmological parameters
        rho = 2.775e11 * omega_m * h**2
        mass = rho * Lbox**3 / N**3
        
        # Grid setup
        Dgrid = Lbox / N
        x0 = 0.5 * Dgrid
        x1 = Lbox - 0.5 * Dgrid
        
        # For simplicity, assume serial execution (no MPI sharding)
        y0 = x0
        y1 = x1
        
        q1d = jnp.linspace(x0, x1, N)
        q1dy = jnp.linspace(y0, y1, N)
        
        qx, qy, qz = jnp.meshgrid(q1d, q1dy, q1d, indexing='ij')
        
        # Interpolate growth factors to redshift z
        D1 = jnp.interp(z, growth_factors.z, growth_factors.d1)
        D2 = jnp.interp(z, growth_factors.z, growth_factors.d2)
        H = jnp.interp(z, growth_factors.z, growth_factors.hubble)
        f1 = jnp.interp(z, growth_factors.z, growth_factors.f1)
        f2 = jnp.interp(z, growth_factors.z, growth_factors.f2)
        
        # Compute positions
        x = qx + D1 * s1x
        y = qy + D1 * s1y
        z = qz + D1 * s1z
        
        if s2x is not None:
            x += D2 * s2x
            y += D2 * s2y
            z += D2 * s2z
        
        # Compute velocities
        vx = a * H * (f1 * s1x)
        vy = a * H * (f1 * s1y)
        vz = a * H * (f1 * s1z)
        
        if s2x is not None:
            vx += a * H * (f2 * D2 * s2x)
            vy += a * H * (f2 * D2 * s2y)
            vz += a * H * (f2 * D2 * s2z)
        
        # Create mass array
        masses = jnp.full_like(x, mass)
        
        return ParticleData(
            positions=jnp.array([x, y, z]),
            velocities=jnp.array([vx, vy, vz]),
            masses=masses
        )
