import logging
log = logging.getLogger(__name__)

def _read_displacement(filename, chunk_shape, chunk_offset):
    import numpy as np
    return np.fromfile(filename, count=chunk_shape[0] * chunk_shape[1] * chunk_shape[2], offset=chunk_offset, dtype=np.float32)

def _profiletime(task_tag, step, times):

    from time import time
    dt = time() - times['t0']
    log.usky_debug(f'{task_tag}: {dt:.6f} sec for {step}')
    if step in times.keys():
        times[step] += dt
    else:
        times[step] = dt
    times['t0'] = time()
    return times

def _sortdict(dictin,reverse=False):
    return dict(sorted(dictin.items(), key=lambda item: item[1], reverse=reverse))

def _summarizetime(task_tag, times):
    total_time = 0
    for key in _sortdict(times,reverse=True).keys():
        if key != 't0':
            log.usky_info(f'{task_tag}: {times[key]:.5e} {key}')
            total_time += times[key]
    log.usky_info(f'{task_tag}: {total_time:.5e} all steps')

class LibField():

    def __init__(self, cosmo_workspace, grid_nside, map_nside, box_length_in_Mpc, zmin, zmax):
        import healpy as hp
        import jax
        import jax.numpy as jnp
        from functools import partial

        self.grid_nside = grid_nside 
        self.map_nside = map_nside
        self.npix = hp.nside2npix(self.map_nside)
        self.L_box = box_length_in_Mpc
        self.cosmo = cosmo_workspace
        self.chimin = self.cosmo.comoving_distance(zmin)
        self.chimax = self.cosmo.comoving_distance(zmax)

        @partial(jax.jit, static_argnames=['trans_vec', 'Dgrid_in_Mpc'])
        def lagrange_mesh(x_axis, y_axis, z_axis, trans_vec, Dgrid_in_Mpc):
            qx, qy, qz = jnp.meshgrid( (x_axis + 0.5 + trans_vec[0]) * Dgrid_in_Mpc, (y_axis + 0.5 + trans_vec[1]) * Dgrid_in_Mpc, (z_axis + 0.5 + trans_vec[2]) * Dgrid_in_Mpc, indexing='ij')
            return qx.ravel(), qy.ravel(), qz.ravel()
        self.lagrange_mesh = lagrange_mesh

        @jax.jit
        def comoving_q(x_i, y_i, z_i):
            return jnp.sqrt(x_i**2. + y_i**2. + z_i**2.)
        self.comoving_q = comoving_q


        # Definitions used for LPT:
        #   grad.S^(n) = - delta^(n)
        # where
        #   delta^(1) = linear density contrast
        #   delta^(2) = Sum [ dSi/dqi * dSj/dqj - (dSi/dqj)^2]
        #   x(q) = q + D * S^(1) + f * D^2 * S^(2)
        # with
        #   f = + 3/7 Omegam_m^(-1/143)
        # being a good approximation for a flat universe
        @partial(jax.jit, static_argnames=['Omega_m'])
        def euclid_i(q_i, s1_i, s2_i, growth_i, Omega_m):
            return (q_i + growth_i * s1_i + (3/7*Omega_m**(-1/143)*growth_i**2.) * s2_i)
        self.euclid_i = euclid_i

    def grid2map(self, s1x, s1y, s1z, s2x, s2y, s2z, grid_xstarts, grid_xstops, grid_ystarts, grid_ystops, grid_zstarts, grid_zstops, nlpt, backend=None):

        import jax
        import jax.numpy as jnp
        from . import jax_healpix as jhp
        from time import time
        from . import kernel_lib  as kl

        tgridmap0 = time()
        overalltimes = {}
        times = {}
        overalltimes={'t0' : time()}
        times={'t0' : time()}
        log = logging.getLogger(__name__)

        task_tag0 = ""
        if backend is not None:
            task_tag0 = backend.jax_backend.task_tag
        task_tag = task_tag0

        Omega_m = self.cosmo.params['Omega_m']

        # Lattice spacing (a_latt in Websky parlance) in Mpc
        lattice_size_in_Mpc = self.L_box / self.grid_nside

        solidang_pix = 4*jnp.pi / self.npix

        # Effectively \Delta chi, comoving distance interval spacing for LoS integral
        geometric_factor = lattice_size_in_Mpc**3. / solidang_pix
        times = _profiletime(task_tag, 'initialization', times)

        # Setup axes for the slab grid
        xaxis = jnp.arange(grid_xstarts, grid_xstops, dtype=jnp.int16)
        yaxis = jnp.arange(grid_ystarts, grid_ystops, dtype=jnp.int16)
        zaxis = jnp.arange(grid_zstarts, grid_zstops, dtype=jnp.int16)
        times = _profiletime(task_tag, 'slab grid axis setup', times)

        log.usky_info(f"grid_xstarts, grid_xstops, grid_ystarts, grid_ystops, grid_zstarts, grid_zstops:"+
                     f"\n    {grid_xstarts}, {grid_xstops}, {grid_ystarts}, {grid_ystops}, {grid_zstarts}, {grid_zstops}")

        skymap = jnp.zeros((self.npix,))
        times = _profiletime(task_tag, 'skymap init', times)

        shift_param = self.grid_nside
        origin_shift = [(0,0,0), (-shift_param,0,0), (0,-shift_param,0), (-shift_param,-shift_param,0),
                        (0,0,-shift_param), (-shift_param,0,-shift_param), (0,-shift_param,-shift_param), (-shift_param,-shift_param,-shift_param)]
        times = _profiletime(task_tag, 'origin shift', times)

        t0 = time()
        for translation in origin_shift:

            # Lagrangian coordinates
            qx, qy, qz = self.lagrange_mesh(xaxis, yaxis, zaxis, translation, lattice_size_in_Mpc)
            times = _profiletime(task_tag, 'Lagrangian meshgrid', times)

            # comoving distance
            chi = jax.vmap(self.comoving_q, in_axes=(0, 0, 0), out_axes=0)(qx, qy, qz)    # 4 : 22
            times = _profiletime(task_tag, 'chi', times)

            # redshift
            redshift = jax.vmap(self.cosmo.comoving_distance2z)(chi)
            times = _profiletime(task_tag, 'redshift', times)

            # healpix indices
            ipix = jhp.vec2pix(self.map_nside, qz, qy, qx)
            times = _profiletime(task_tag, 'ipix', times)

            # lensing kernel
            kernel = -jnp.where((chi >= self.chimin) & (chi <= self.chimax), jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )(self.cosmo, geometric_factor, chi, redshift), 0.)
            times = _profiletime(task_tag, 'kernel', times)

            # add lensing kernel to corresponding skymap pixel at each grid position
            skymap = skymap.at[ipix].add(kernel)
            times = _profiletime(task_tag, 'skymap add', times)

            del kernel, ipix
            times = _profiletime(task_tag, 'delete kernel, ipix', times)

            # linear growth factor
            growth = jax.vmap(self.cosmo.growth_factor_D)(redshift)
            times = _profiletime(task_tag, 'growth', times)

            # Eulerian x coordinate
            if nlpt == 1: Xx = jax.vmap(self.euclid_i, in_axes=(0, 0, None, 0, None), out_axes=0)(qx, s1x, 0., growth, Omega_m)
            if nlpt == 2: Xx = jax.vmap(self.euclid_i, in_axes=(0, 0, 0, 0, None), out_axes=0)(qx, s1x, s2x, growth, Omega_m)
                
            times = _profiletime(task_tag, 'Xx', times)

            del qx
            times = _profiletime(task_tag, 'qx delete', times)

            # Eulerian y coordinate
            if nlpt == 1: Xy = jax.vmap(self.euclid_i, in_axes=(0, 0, None, 0, None), out_axes=0)(qy, s1y, 0., growth, Omega_m)
            if nlpt == 2: Xy = jax.vmap(self.euclid_i, in_axes=(0, 0, 0, 0, None), out_axes=0)(qy, s1y, s2y, growth, Omega_m)
            times = _profiletime(task_tag, 'Xy', times)

            del qy
            times = _profiletime(task_tag, 'qy delete', times)

            # Eulerian z coordinate
            
            if nlpt == 1: Xz = jax.vmap(self.euclid_i, in_axes=(0, 0, None, 0, None), out_axes=0)(qz, s1z, 0., growth, Omega_m)
            if nlpt == 2: Xz = jax.vmap(self.euclid_i, in_axes=(0, 0, 0, 0, None), out_axes=0)(qz, s1z, s2z, growth, Omega_m)
            
                
            times = _profiletime(task_tag, 'Xz', times)

            del qz, growth
            times = _profiletime(task_tag, 'qz, growth delete', times)

            ipix = jhp.vec2pix(self.map_nside, Xz, Xy, Xx)
            times = _profiletime(task_tag, 'ipix Eulerian', times)

            del Xx, Xy, Xz
            times = _profiletime(task_tag, 'Xx, Xy, Xz delete', times)

            kernel = jnp.where((chi >= self.chimin) & (chi <= self.chimax),
                               jax.vmap(kl.lensing_kernel_F, in_axes=(None, None, 0, 0), out_axes=0 )
                                       (self.cosmo, geometric_factor, chi, redshift), 0.)
            times = _profiletime(task_tag, 'kernel Eulerian', times)

            del chi, redshift
            times = _profiletime(task_tag, 'chi, redshift delete Eulerian', times)

            skymap = skymap.at[ipix].add(kernel)
            times = _profiletime(task_tag, 'skymap add Eulerian', times)

            del ipix, kernel
            times = _profiletime(task_tag, 'ipix, kernel delete Eulerian', times)

        del s1x, s1y, s1z, s2x, s2y, s2z
        times = _profiletime(task_tag+' (grid2map)', 'sx, sy, sz delete', times)
        _summarizetime(task_tag+' (grid2map steps)',times)

        overalltimes = _profiletime(task_tag, 'grid2map', overalltimes)
        _summarizetime(task_tag+' (grid2map)',overalltimes)

        return skymap
    
    def fieldmap(self, displacements, backend, nlpt=2, is64bit=False, use_tqdm=False, peak_per_cell_memory_in_MB = 200.0, jax_overhead_factor  = 1.5):        #kernel_list,
        
        import jax
        import jax.numpy as jnp
        import numpy as np
        from time import time

        data_shape = (self.grid_nside, self.grid_nside, self.grid_nside)
        bytes_per_cell_of_data = 4.
        if is64bit: bytes_per_cell_of_data = 8.

        # print(is64bit, bytes_per_cell_of_data, peak_per_cell_memory_in_MB)

        backend.datastream_setup(data_shape, bytes_per_cell_of_data, peak_per_cell_memory_in_MB, jax_overhead_factor, decom_type='slab', divide_axis=0)
        jax_iterator = backend.get_iterator()
        obs_map = np.zeros((self.npix,))
        task_tag = backend.jax_backend.task_tag
        iterator = jax_iterator

        if displacements['type'] == 'arraylist':
            s1x = displacements['data'][0]
            s1y = displacements['data'][1]
            s1z = displacements['data'][2]

            if nlpt == 2:
                s2x = displacements['data'][3]
                s2y = displacements['data'][4]
                s2z = displacements['data'][5]
            else:
                s2x = None
                s2y = None
                s2z = None

            # for array list of displacements, domain decomposition already done (currently with jax sharding)
            # redefine the iterator to one element containing stop and start in global array
            iterator = [{'start' : displacements['start'], 'stop' : displacements['stop'], 'offset' : None, 'shape' : None}]

        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(jax_iterator, ncols=120)
        else:
            i=0 ; t=0. ; tbar=0. ; tread=0. ; tmap=0.
            n=len(iterator)
        for iter in iterator:

            log.usky_debug(f"{ iter }", per_task=True)

            start  = iter['start']
            stop   = iter['stop']
            offset = iter['offset']
            shape  = iter['shape']

            if not use_tqdm:
                t1=time()
            if displacements['type'] == 'filelist':
                s1x = _read_displacement(displacements['data'][0], shape, offset)
                s1y = _read_displacement(displacements['data'][1], shape, offset)
                s1z = _read_displacement(displacements['data'][2], shape, offset)

                if nlpt == 2:
                    s2x = _read_displacement(displacements['data'][3], shape, offset)
                    s2y = _read_displacement(displacements['data'][4], shape, offset)
                    s2z = _read_displacement(displacements['data'][5], shape, offset)

                startx = iter['start'] ; stopx = iter['stop']
                starty = 0             ; stopy = self.grid_nside
                startz = 0             ; stopz = self.grid_nside
            elif displacements['type'] == 'arraylist':
                # for array list of displacements, domain decomposition already done (currently with jax sharding)
                # and encoded in the iterator from displacements['start']['x'], etc.
                startx = iter['start']['x'] ; stopx = iter['stop']['x']
                starty = iter['start']['y'] ; stopy = iter['stop']['y']
                startz = iter['start']['z'] ; stopz = iter['stop']['z']

            if not use_tqdm:
                t2=time()

            times={'t0' : time()}

            s1x = jnp.asarray(s1x) ; s1y = jnp.asarray(s1y) ; s1z = jnp.asarray(s1z)
            times = _profiletime(task_tag, 'numpy to jax s1x, s1y, s1z', times)

            s2x = jnp.asarray(s2x) ; s2y = jnp.asarray(s2y) ; s2z = jnp.asarray(s2z)
            times = _profiletime(task_tag, 'numpy to jax s2x, s2y, s2z', times)

            obs_map_cur = self.grid2map(s1x, s1y, s1z, s2x, s2y, s2z, startx, stopx, starty, stopy, startz, stopz, nlpt, backend=backend)
            times = _profiletime(task_tag, 'grid2map in fieldmap', times)

            obs_map_cur = np.array(obs_map_cur, dtype=np.float32)
            times = _profiletime(task_tag, 'jax to numpy obs_map', times)

            obs_map += obs_map_cur  #, kernel_list
            times = _profiletime(task_tag, 'accumulate obs_map', times)

            _summarizetime(task_tag+' (fieldmap mapmaking)', times)

            if not use_tqdm:
                t3=time()
                i += 1
                dtread = t2-t1
                dtmap  = t3-t2
                tread += dtread
                tmap  += dtmap
                tread_bar = tread / i
                tmap_bar  = tmap  / i

        return backend.mpi_backend.reduce2map(obs_map)
    




            