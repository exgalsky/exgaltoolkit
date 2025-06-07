"""
Initial conditions writer for various formats.
"""
import os
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Any, List

class ICWriter:
    """
    Writes initial conditions to various formats.
    
    This replaces the ICs class with a cleaner interface focused on writing
    initial conditions data to different simulation formats.
    """
    
    def __init__(self, ic_generator, output_dir: str = "./output"):
        """
        Initialize the IC writer.
        
        Parameters:
        -----------
        ic_generator : ICGenerator
            The IC generator containing the data to write
        output_dir : str
            Output directory for files
        """
        self.ic_generator = ic_generator
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get cosmological parameters
        self.cosmology = ic_generator.cosmology_service.parameters
        self.h = self.cosmology.h
        self.omegam = self.cosmology.Omega_m
        
        # Get simulation parameters
        self.N = ic_generator.N
        self.Lbox = ic_generator.Lbox
        self.z_initial = ic_generator.z_initial
        
        # Get parallel info
        self.comm = ic_generator.comm
        self.mpiproc = ic_generator.mpiproc
        self.nproc = ic_generator.nproc
        
    def write_initial_conditions(self, 
                                filename: Optional[str] = None,
                                format: str = 'nyx') -> List[str]:
        """
        Write initial conditions to file.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename. If None, uses default naming.
        format : str
            Output format. Currently supports 'nyx'.
            
        Returns:
        --------
        output_files : list
            List of output filenames created
        """
        if filename is None:
            filename = f"ic_N{self.N}_z{self.z_initial:.1f}"
        
        if format == 'nyx':
            return self._write_nyx_format(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _write_nyx_format(self, base_filename: str) -> List[str]:
        """
        Write initial conditions in Nyx format.
        
        Nyx format specification:
        - Header: npart (long), ndim (int), nx (int)
        - Data: x, y, z, mass, vx, vy, vz (all float32)
        
        Units:
        - Position: Mpc
        - Velocity: km/s (peculiar proper velocity)
        - Mass: M☉
        """
        # Get particle data
        positions = self.ic_generator.get_particle_positions()
        velocities = self.ic_generator.get_particle_velocities()
        
        if positions is None or velocities is None:
            raise ValueError("Particle positions and velocities must be computed first")
        
        # Calculate particle mass
        rho_crit = 2.775e11 * self.omegam * self.h**2  # Critical density in M☉/Mpc³
        mass_per_particle = rho_crit * self.Lbox**3 / self.N**3
        
        if self.mpiproc == 0:
            print(f"Particle mass: {mass_per_particle:.3e} M☉")
            print(f"Box size: {self.Lbox} Mpc")
        
        # Get local particle count
        local_npart = positions.shape[0] * positions.shape[1] * positions.shape[2]
        
        # Extract coordinates
        x = positions[..., 0].flatten()
        y = positions[..., 1].flatten()
        z = positions[..., 2].flatten()
        
        vx = velocities[..., 0].flatten()
        vy = velocities[..., 1].flatten()
        vz = velocities[..., 2].flatten()
        
        # Create mass array
        mass = np.full(local_npart, mass_per_particle, dtype=np.float32)
        
        # Write file
        if self.nproc > 1:
            # MPI case - each process writes its own file
            filename = f"{base_filename}_rank{self.mpiproc:04d}.nyx"
        else:
            filename = f"{base_filename}.nyx"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as fid:
            # Write header
            np.array([local_npart], dtype='int64').tofile(fid)  # npart (long)
            np.array([3], dtype='int32').tofile(fid)            # ndim (3D)
            np.array([4], dtype='int32').tofile(fid)            # nx (4 extra fields: mass + 3 velocities)
            
            # Write particle data
            # Nyx expects: x, y, z, mass, vx, vy, vz for each particle
            particle_data = np.column_stack([
                x.astype(np.float32),
                y.astype(np.float32),
                z.astype(np.float32),
                mass,
                vx.astype(np.float32),
                vy.astype(np.float32),
                vz.astype(np.float32)
            ])
            
            particle_data.tofile(fid)
        
        if self.mpiproc == 0:
            print(f"Written initial conditions to {filepath}")
        
        return [filepath]
    
    def write_ascii_format(self, filename: str) -> List[str]:
        """
        Write initial conditions in ASCII format for debugging.
        
        Parameters:
        -----------
        filename : str
            Base filename for output
            
        Returns:
        --------
        output_files : list
            List of output filenames created
        """
        positions = self.ic_generator.get_particle_positions()
        velocities = self.ic_generator.get_particle_velocities()
        
        if positions is None or velocities is None:
            raise ValueError("Particle positions and velocities must be computed first")
        
        # Calculate particle mass
        rho_crit = 2.775e11 * self.omegam * self.h**2
        mass_per_particle = rho_crit * self.Lbox**3 / self.N**3
        
        if self.nproc > 1:
            filename = f"{filename}_rank{self.mpiproc:04d}.txt"
        else:
            filename = f"{filename}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"# Initial conditions at z = {self.z_initial}\n")
            f.write(f"# Box size: {self.Lbox} Mpc\n")
            f.write(f"# Grid size: {self.N}^3\n")
            f.write(f"# Particle mass: {mass_per_particle:.3e} M☉\n")
            f.write(f"# Columns: x y z vx vy vz\n")
            
            # Flatten arrays for writing
            pos_flat = positions.reshape(-1, 3)
            vel_flat = velocities.reshape(-1, 3)
            
            for i in range(pos_flat.shape[0]):
                f.write(f"{pos_flat[i, 0]:.6f} {pos_flat[i, 1]:.6f} {pos_flat[i, 2]:.6f} "
                       f"{vel_flat[i, 0]:.6f} {vel_flat[i, 1]:.6f} {vel_flat[i, 2]:.6f}\n")
        
        if self.mpiproc == 0:
            print(f"Written ASCII initial conditions to {filepath}")
        
        return [filepath]
    
    def write_statistics(self, filename: str) -> List[str]:
        """
        Write statistics about the initial conditions.
        
        Parameters:
        -----------
        filename : str
            Base filename for statistics output
            
        Returns:
        --------
        output_files : list
            List of output filenames created
        """
        # Only rank 0 writes statistics
        if self.mpiproc != 0:
            return []
        
        filepath = os.path.join(self.output_dir, f"{filename}_stats.txt")
        
        with open(filepath, 'w') as f:
            f.write("=== Initial Conditions Statistics ===\n\n")
            
            # Simulation parameters
            f.write("Simulation Parameters:\n")
            f.write(f"  Grid size: {self.N}^3\n")
            f.write(f"  Box size: {self.Lbox} Mpc\n")
            f.write(f"  Initial redshift: {self.z_initial}\n")
            f.write(f"  LPT order: {self.ic_generator.lpt_order}\n")
            f.write(f"  Random seed: {self.ic_generator.seed}\n")
            f.write(f"  MPI processes: {self.nproc}\n\n")
            
            # Cosmological parameters
            f.write("Cosmological Parameters:\n")
            f.write(f"  h: {self.h}\n")
            f.write(f"  Omega_m: {self.omegam}\n")
            f.write(f"  Omega_b: {self.cosmology.Omega_b}\n")
            f.write(f"  n_s: {self.cosmology.n_s}\n")
            f.write(f"  sigma_8: {self.cosmology.sigma_8}\n\n")
            
            # Field statistics
            if hasattr(self.ic_generator, '_last_results'):
                results = self.ic_generator._last_results
                
                if 'delta_stats' in results:
                    delta_stats = results['delta_stats']
                    f.write("Density Field Statistics:\n")
                    f.write(f"  Mean: {delta_stats.get('mean', 'N/A'):.6f}\n")
                    f.write(f"  Std: {delta_stats.get('std', 'N/A'):.6f}\n")
                    f.write(f"  Min: {delta_stats.get('min', 'N/A'):.6f}\n")
                    f.write(f"  Max: {delta_stats.get('max', 'N/A'):.6f}\n\n")
                
                if 'particle_stats' in results:
                    particle_stats = results['particle_stats']
                    f.write("Particle Statistics:\n")
                    if 'positions' in particle_stats:
                        pos_stats = particle_stats['positions']
                        f.write(f"  Position shape: {pos_stats.get('shape', 'N/A')}\n")
                        f.write(f"  X range: {pos_stats.get('x_range', 'N/A')}\n")
                        f.write(f"  Y range: {pos_stats.get('y_range', 'N/A')}\n")
                        f.write(f"  Z range: {pos_stats.get('z_range', 'N/A')}\n")
                    
                    if 'velocities' in particle_stats:
                        vel_stats = particle_stats['velocities']
                        f.write(f"  Velocity shape: {vel_stats.get('shape', 'N/A')}\n")
                        f.write(f"  RMS velocity: {vel_stats.get('rms_velocity', 'N/A'):.6f} km/s\n")
        
        print(f"Written statistics to {filepath}")
        return [filepath]
