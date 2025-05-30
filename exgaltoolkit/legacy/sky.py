"""
Legacy Sky class wrapper for backward compatibility.
"""
from typing import Optional, Dict, Any
import warnings

from ..api.factory import SimulationFactory
from ..api.simulation import MockGenerationSimulation
from ..core.config import SimulationConfig


class Sky:
    """Legacy Sky class wrapper for backward compatibility."""
    
    def __init__(self, **kwargs):
        """Initialize Sky with legacy interface."""
        # Store original kwargs for reference
        self._kwargs = kwargs
        
        # Extract known parameters with defaults (matching original sky.py)
        self.ID = kwargs.get('ID', 'MockgenDefaultID')
        self.seed = kwargs.get('seed', self.ID)
        self.N = kwargs.get('N', 128)
        self.Niter = kwargs.get('Niter', 1)
        self.input = kwargs.get('input', 'delta')
        self.Lbox = kwargs.get('Lbox', 7700.0)
        self.zInit = kwargs.get('zInit', 99.0)
        self.laststep = kwargs.get('laststep', 'writeics')
        self.Nside = kwargs.get('Nside', 512)
        self.icw = kwargs.get('icw', True)
        self.nlpt = kwargs.get('nlpt', 2)
        self.gpu = kwargs.get('gpu', False)
        self.mpi = kwargs.get('mpi', True)
        
        # Initialize MPI attributes (for compatibility)
        from mpi4py import MPI
        self.parallel = False
        self.nproc = MPI.COMM_WORLD.Get_size()
        self.mpiproc = MPI.COMM_WORLD.Get_rank()
        self.comm = MPI.COMM_WORLD
        self.task_tag = f"MPI process {self.mpiproc}"
        
        if MPI.COMM_WORLD.Get_size() > 1:
            self.parallel = True
            
        # Handle cosmology interface
        self.cosmo = kwargs.get('cosmo', None)
        if self.cosmo is None:
            # Create default cosmology
            from .cosmo import CosmologyInterface
            self.cosmo = CosmologyInterface()
            
        # Initialize cube to None (will be created in run)
        self.cube = None
        
        # Create internal simulation
        self._simulation = None
        
    def run(self, **kwargs):
        """Run simulation with legacy interface."""
        import jax
        from time import time
        
        # Set up timing
        times = {'t0': time()}
        
        # Create simulation if not exists
        if self._simulation is None:
            try:
                self._simulation = SimulationFactory.from_legacy_kwargs(**self._kwargs)
            except Exception as e:
                # Fall back to original implementation if new one fails
                warnings.warn(f"New implementation failed, falling back to legacy: {e}")
                return self._run_legacy(**kwargs)
        
        # Initialize cube for compatibility
        if not self.parallel:
            from .cube import Cube
            self.cube = Cube(N=self.N, Lbox=self.Lbox, partype=None)
        else:
            import exgaltoolkit.util.jax_util as ju
            ju.distributed_initialize()
            from .cube import Cube
            self.cube = Cube(N=self.N, Lbox=self.Lbox)
            
        if self.laststep == 'init':
            return 0
            
        err = 0
        seeds = range(self.seed, self.seed + self.Niter)
        i = 0
        for seed in seeds:
            if i == 1:
                times = {'t0': time()}
            err += self.generatesky(seed, times)
            i += 1
            
        import exgaltoolkit.util.log_util as xglogutil
        xglogutil.summarizetime(None, times, self.comm, self.mpiproc)
        
        return err
    
    def generatesky(self, seed, times, **kwargs):
        """Generate sky with specific seed (legacy interface)."""
        if self._simulation is None:
            # Fall back to legacy implementation
            return self._generatesky_legacy(seed, times, **kwargs)
            
        try:
            # Run new implementation
            # Update seed in configuration
            self._simulation.config.simulation.seed = seed
            
            # Run simulation
            results = self._simulation.run()
            
            # Update timing and logging for compatibility
            import exgaltoolkit.util.log_util as xglogutil
            if self.mpiproc == 0:
                xglogutil.parprint(f'\\nGenerating sky for model "{self.ID}" with seed={seed}')
                
            times = xglogutil.profiletime(None, 'noise generation', times, self.comm, self.mpiproc)
            if self.laststep == 'noise':
                return 0
                
            times = xglogutil.profiletime(None, 'noise convolution', times, self.comm, self.mpiproc)
            if self.laststep == 'convolution':
                return 0
                
            times = xglogutil.profiletime(None, 'LPT', times, self.comm, self.mpiproc)
            if self.laststep == 'LPT':
                return 0
                
            if self.icw:
                times = xglogutil.profiletime(None, 'write ICs', times, self.comm, self.mpiproc)
            if self.laststep == 'writeics':
                return 0
                
            return 0
            
        except Exception as e:
            warnings.warn(f"New implementation failed for seed {seed}, falling back to legacy: {e}")
            return self._generatesky_legacy(seed, times, **kwargs)
    
    def _run_legacy(self, **kwargs):
        """Fall back to original implementation."""
        # Import original implementation
        from ..mockgen.sky import Sky as OriginalSky
        
        # Create original sky object
        original_sky = OriginalSky(**self._kwargs)
        
        # Copy relevant attributes
        self.cube = original_sky.cube
        self.cosmo = original_sky.cosmo
        
        # Run original implementation
        return original_sky.run(**kwargs)
    
    def _generatesky_legacy(self, seed, times, **kwargs):
        """Fall back to original generatesky implementation."""
        from ..mockgen.sky import Sky as OriginalSky
        
        # Create original sky object if needed
        if not hasattr(self, '_original_sky'):
            self._original_sky = OriginalSky(**self._kwargs)
            self.cube = self._original_sky.cube
            self.cosmo = self._original_sky.cosmo
            
        return self._original_sky.generatesky(seed, times, **kwargs)
