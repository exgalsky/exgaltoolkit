"""
Main initial conditions generator.
"""
import logging
from typing import Optional, Dict, Any
from time import time
import exgaltoolkit.util.log_util as xglogutil
import exgaltoolkit.util.backend as xgback
import exgaltoolkit.util.jax_util as ju
from ..cosmology import CosmologyService, CosmologicalParameters
from ..grid import GridOperations, LPTCalculator
from .writer import ICWriter

class ICGenerator:
    """
    Main initial conditions generator.
    
    This replaces the Sky class with a cleaner, more focused interface
    for generating cosmological initial conditions.
    """
    
    def __init__(self, 
                 N: int = 128,
                 Lbox: float = 7700.0,
                 cosmology: Optional[CosmologicalParameters] = None,
                 seed: int = 12345,
                 lpt_order: int = 2,
                 z_initial: float = 99.0,
                 output_dir: str = "./output",
                 **kwargs):
        """
        Initialize the initial conditions generator.
        
        Parameters:
        -----------
        N : int
            Grid size (N^3 particles)
        Lbox : float
            Box size in Mpc/h
        cosmology : CosmologicalParameters, optional
            Cosmological parameters. If None, uses defaults.
        seed : int
            Random seed for reproducibility
        lpt_order : int
            LPT order (1 or 2)
        z_initial : float
            Initial redshift
        output_dir : str
            Output directory for files
        """
        self.N = N
        self.Lbox = Lbox
        self.seed = seed
        self.lpt_order = lpt_order
        self.z_initial = z_initial
        self.output_dir = output_dir
        
        # Set up cosmology
        if cosmology is None:
            cosmology = CosmologicalParameters()
        
        # Generate power spectrum using CAMB if possible
        power_spectrum = kwargs.get('power_spectrum', None)
        if power_spectrum is None:
            power_spectrum = self._generate_camb_power_spectrum(cosmology, z_initial)
        
        self.cosmology_service = CosmologyService(cosmology, power_spectrum)
        
        # Set up grid operations
        self.grid_ops = GridOperations(N=N, Lbox=Lbox, **kwargs)
        self.lpt_calc = LPTCalculator(self.grid_ops, order=lpt_order)
        
        # Set up MPI if available
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpiproc = MPI.COMM_WORLD.Get_rank()
            self.nproc = MPI.COMM_WORLD.Get_size()
            self.parallel = self.nproc > 1
        except ImportError:
            self.comm = None
            self.mpiproc = 0
            self.nproc = 1
            self.parallel = False
        
        # Track state
        self._noise_generated = False
        self._delta_computed = False
        self._lpt_computed = False
        
        # Results
        self.particle_positions = None
        self.particle_velocities = None
    
    def generate_initial_conditions(self, 
                                   save_output: bool = True,
                                   output_filename: Optional[str] = None,
                                   steps: Optional[list] = None) -> Dict[str, Any]:
        """
        Generate complete initial conditions.
        
        Parameters:
        -----------
        save_output : bool
            Whether to save output files
        output_filename : str, optional
            Custom output filename
        steps : list, optional
            List of steps to run. If None, runs all steps.
            Options: ['noise', 'delta', 'lpt', 'particles', 'output']
            
        Returns:
        --------
        results : dict
            Dictionary containing results and metadata
        """
        if steps is None:
            steps = ['noise', 'delta', 'lpt', 'particles']
            if save_output:
                steps.append('output')
        
        times = {'t0': time()}
        
        # Initialize distributed computing if needed
        if self.parallel:
            ju.distributed_initialize()
        
        results = {}
        
        try:
            # Step 1: Generate noise
            if 'noise' in steps:
                if self.mpiproc == 0:
                    xglogutil.parprint(f'Generating noise field with seed={self.seed}')
                
                self.grid_ops.generate_noise(seed=self.seed)
                self._noise_generated = True
                times = xglogutil.profiletime(None, 'noise generation', times, self.comm, self.mpiproc)
                
                results['noise_generated'] = True
            
            # Step 2: Convolve to get density field
            if 'delta' in steps and self._noise_generated:
                if self.mpiproc == 0:
                    xglogutil.parprint('Computing density field from noise')
                
                # Set up backend for this step
                backend = xgback.Backend(force_no_gpu=True, force_no_mpi=True, 
                                       logging_level=-logging.ERROR)
                
                self.grid_ops.convolve_with_transfer_function(self.cosmology_service)
                self._delta_computed = True
                times = xglogutil.profiletime(None, 'density field computation', times, self.comm, self.mpiproc)
                
                results['delta_computed'] = True
                results['delta_stats'] = self._compute_field_stats(self.grid_ops.get_density_field())
            
            # Step 3: Compute LPT displacements
            if 'lpt' in steps and self._delta_computed:
                if self.mpiproc == 0:
                    xglogutil.parprint(f'Computing LPT displacements (order {self.lpt_order})')
                
                self.lpt_calc.compute_displacements(input_mode='noise')
                self._lpt_computed = True
                times = xglogutil.profiletime(None, 'LPT computation', times, self.comm, self.mpiproc)
                
                results['lpt_computed'] = True
                displacements = self.grid_ops.get_displacement_fields()
                results['displacement_stats'] = self._compute_displacement_stats(displacements)
            
            # Step 4: Generate particle positions and velocities
            if 'particles' in steps and self._lpt_computed:
                if self.mpiproc == 0:
                    xglogutil.parprint('Computing particle positions and velocities')
                
                self.particle_positions = self.lpt_calc.compute_particle_positions()
                self.particle_velocities = self.lpt_calc.compute_particle_velocities(
                    self.cosmology_service, self.z_initial
                )
                times = xglogutil.profiletime(None, 'particle computation', times, self.comm, self.mpiproc)
                
                results['particles_computed'] = True
                results['particle_stats'] = self._compute_particle_stats()
            
            # Step 5: Save output
            if 'output' in steps and save_output:
                if self.mpiproc == 0:
                    xglogutil.parprint('Writing initial conditions files')
                
                writer = ICWriter(self, output_dir=self.output_dir)
                output_files = writer.write_initial_conditions(filename=output_filename)
                times = xglogutil.profiletime(None, 'output writing', times, self.comm, self.mpiproc)
                
                results['output_files'] = output_files
            
            # Summarize timing
            xglogutil.summarizetime(None, times, self.comm, self.mpiproc)
            
            results['success'] = True
            results['total_time'] = time() - times['t0']
            
        except Exception as e:
            if self.mpiproc == 0:
                print(f"Error in initial conditions generation: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def _compute_field_stats(self, field) -> Dict[str, float]:
        """Compute statistics for a field."""
        if field is None:
            return {}
        
        import jax.numpy as jnp
        return {
            'mean': float(jnp.mean(field)),
            'std': float(jnp.std(field)),
            'min': float(jnp.min(field)),
            'max': float(jnp.max(field))
        }
    
    def _compute_displacement_stats(self, displacements) -> Dict[str, Dict[str, float]]:
        """Compute statistics for displacement fields."""
        s1x, s1y, s1z, s2x, s2y, s2z = displacements
        stats = {}
        
        if s1x is not None:
            stats['s1x'] = self._compute_field_stats(s1x)
            stats['s1y'] = self._compute_field_stats(s1y)
            stats['s1z'] = self._compute_field_stats(s1z)
        
        if s2x is not None:
            stats['s2x'] = self._compute_field_stats(s2x)
            stats['s2y'] = self._compute_field_stats(s2y)
            stats['s2z'] = self._compute_field_stats(s2z)
        
        return stats
    
    def _compute_particle_stats(self) -> Dict[str, Any]:
        """Compute statistics for particles."""
        stats = {}
        
        if self.particle_positions is not None:
            import jax.numpy as jnp
            pos = self.particle_positions
            stats['positions'] = {
                'shape': pos.shape,
                'x_range': [float(jnp.min(pos[..., 0])), float(jnp.max(pos[..., 0]))],
                'y_range': [float(jnp.min(pos[..., 1])), float(jnp.max(pos[..., 1]))],
                'z_range': [float(jnp.min(pos[..., 2])), float(jnp.max(pos[..., 2]))]
            }
        
        if self.particle_velocities is not None:
            import jax.numpy as jnp
            vel = self.particle_velocities
            rms_vel = float(jnp.sqrt(jnp.mean(vel**2)))
            stats['velocities'] = {
                'shape': vel.shape,
                'rms_velocity': rms_vel
            }
        
        return stats
    
    def _generate_camb_power_spectrum(self, cosmology: CosmologicalParameters, z_initial: float) -> dict:
        """Generate power spectrum using CAMB."""
        try:
            import camb
            import jax.numpy as jnp
            
            # Convert our cosmological parameters to CAMB format
            H0 = cosmology.h * 100  # CAMB expects H0 in km/s/Mpc
            
            # Set up CAMB parameters
            camb_par = camb.set_params(
                H0=H0,
                ombh2=cosmology.omega_b * cosmology.h**2,
                omch2=(cosmology.omega_m - cosmology.omega_b) * cosmology.h**2,
                omk=cosmology.omega_k,
                tau=0.066,  # Default optical depth
                As=2e-9,    # Will be scaled by sigma_8
                ns=cosmology.n_s
            )
            
            # Set matter power spectrum
            camb_par.set_matter_power(redshifts=[z_initial], kmax=2.0)
            
            # Get results
            camb_wsp = camb.get_results(camb_par)
            k, zlist, pk = camb_wsp.get_matter_power_spectrum(
                minkh=1e-4, maxkh=1e2, npoints=2000
            )
            
            return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk[0, :])}
            
        except ImportError:
            # CAMB not available, use default power spectrum
            import exgaltoolkit.util.log_util as xglogutil
            if hasattr(self, 'mpiproc') and self.mpiproc == 0:
                xglogutil.parprint("CAMB not available, using default power spectrum")
            return None  # Will use default from CosmologyService
        except Exception as e:
            # CAMB failed, use default
            import exgaltoolkit.util.log_util as xglogutil
            if hasattr(self, 'mpiproc') and self.mpiproc == 0:
                xglogutil.parprint(f"CAMB failed ({e}), using default power spectrum")
            return None

    # Convenience methods for step-by-step execution
    def generate_noise(self):
        """Generate noise field only."""
        return self.generate_initial_conditions(save_output=False, steps=['noise'])
    
    def compute_density_field(self):
        """Compute density field (requires noise)."""
        return self.generate_initial_conditions(save_output=False, steps=['delta'])
    
    def compute_lpt_displacements(self):
        """Compute LPT displacements (requires density field)."""
        return self.generate_initial_conditions(save_output=False, steps=['lpt'])
    
    def compute_particles(self):
        """Compute particle positions and velocities (requires LPT)."""
        return self.generate_initial_conditions(save_output=False, steps=['particles'])
    
    # Getter methods for results
    def get_density_field(self):
        """Get the density contrast field."""
        return self.grid_ops.get_density_field()
    
    def get_displacement_fields(self):
        """Get LPT displacement fields."""
        return self.grid_ops.get_displacement_fields()
    
    def get_particle_positions(self):
        """Get particle positions."""
        return self.particle_positions
    
    def get_particle_velocities(self):
        """Get particle velocities."""
        return self.particle_velocities
