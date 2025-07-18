# example for using a user-provided p(k) made with camb - updated for new API

# Initialize distributed JAX before any imports if running in parallel
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    if nproc > 1:
        import jax
        jax.distributed.initialize()
except ImportError:
    rank = 0
    nproc = 1

import camb
import numpy as np
import jax.numpy as jnp
from exgaltoolkit import ICGenerator, CosmologicalParameters, CosmologyService

zics=100
camb_par = camb.set_params(H0=68)
camb_par.set_matter_power(redshifts=[zics], kmax=2.0)
camb_wsp = camb.get_results(camb_par)

def my_get_pspec():
    k, zlist, pk = camb_wsp.get_matter_power_spectrum(
                  minkh=1e-4, maxkh=1e2, npoints = 2000)
    return {'k': jnp.asarray(k), 'pofk': jnp.asarray(pk[0,:])}

# Create cosmology setup with new API
cosmo_params = CosmologicalParameters(H0=68)
cosmo_service = CosmologyService(cosmo_params, power_spectrum=my_get_pspec())

# Create IC generator with the cosmology service
ic_gen = ICGenerator(
    N=128,
    Lbox=7700.0,  # Box size required in new API
    cosmology=cosmo_params,
    seed=13579,
    lpt_order=1,
    partype='jaxshard' if nproc > 1 else None
)
ic_gen.cosmology_service = cosmo_service

# Generate initial conditions and get density field
result = ic_gen.generate_initial_conditions(save_output=False)
delta = np.asarray(ic_gen.get_density_field())

# Get LPT displacement fields
displacements = ic_gen.get_displacement_fields()
s1x = np.asarray(displacements[0])
s1y = np.asarray(displacements[1])
s1z = np.asarray(displacements[2])

# Debug: Print grid shapes to check if sharding is working
print(f"[Rank {rank}] Delta grid shape: {delta.shape}")
print(f"[Rank {rank}] S1x grid shape: {s1x.shape}")
print(f"[Rank {rank}] Expected shape for 4 processes: (128, 32, 128) or similar slab decomposition")

# Save results
np.savez("./output/grids",
         delta=delta,
         s1x=s1x,
         s1y=s1y,
         s1z=s1z)

# Plot 2x2 panel showing [0,:,:] slice of each grid
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Initial Conditions Grids - Slice [0,:,:] - Rank {rank}/{nproc-1}', fontsize=16)

# Delta field
im1 = axes[0,0].imshow(delta[0,:,:], cmap='RdBu_r', origin='lower')
axes[0,0].set_title('Delta (density field)')
axes[0,0].set_xlabel('y')
axes[0,0].set_ylabel('z')
plt.colorbar(im1, ax=axes[0,0])

# S1x displacement
im2 = axes[0,1].imshow(s1x[0,:,:], cmap='RdBu_r', origin='lower')
axes[0,1].set_title('S1x (x-displacement)')
axes[0,1].set_xlabel('y')
axes[0,1].set_ylabel('z')
plt.colorbar(im2, ax=axes[0,1])

# S1y displacement
im3 = axes[1,0].imshow(s1y[0,:,:], cmap='RdBu_r', origin='lower')
axes[1,0].set_title('S1y (y-displacement)')
axes[1,0].set_xlabel('y')
axes[1,0].set_ylabel('z')
plt.colorbar(im3, ax=axes[1,0])

# S1z displacement
im4 = axes[1,1].imshow(s1z[0,:,:], cmap='RdBu_r', origin='lower')
axes[1,1].set_title('S1z (z-displacement)')
axes[1,1].set_xlabel('y')
axes[1,1].set_ylabel('z')
plt.colorbar(im4, ax=axes[1,1])

plt.tight_layout()
filename = f'./output/grids_visualization_rank{rank:04d}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Visualization saved to {filename}")