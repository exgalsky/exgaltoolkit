def _get_lpt_displacement_files(backend,N):
    import logging
    import os
    log = logging.getLogger("LIGHTCONE")

    # only websky displacement fields are implemented
    try:
        path2disp = os.environ['LPT_DISPLACEMENTS_PATH']
    except:
        path2disp = '/Users/shamik/Documents/Work/websky_datacube/'

    backend.print2log(log, f"Path to displacement files set to {path2disp}", level='usky_info')

    if N == 768:
        s1xfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16_no768'
        s1yfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16_no768'
        s1zfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16_no768'

        s2xfile = path2disp+'sx2_7700Mpc_n6144_nb30_nt16_no768'
        s2yfile = path2disp+'sy2_7700Mpc_n6144_nb30_nt16_no768'
        s2zfile = path2disp+'sz2_7700Mpc_n6144_nb30_nt16_no768'
    else:
        s1xfile = path2disp+'sx1_7700Mpc_n6144_nb30_nt16'
        s1yfile = path2disp+'sy1_7700Mpc_n6144_nb30_nt16'
        s1zfile = path2disp+'sz1_7700Mpc_n6144_nb30_nt16'

        s2xfile = path2disp+'sx2_7700Mpc_n6144_nb30_nt16'
        s2yfile = path2disp+'sy2_7700Mpc_n6144_nb30_nt16'
        s2zfile = path2disp+'sz2_7700Mpc_n6144_nb30_nt16'

    displacements = {}
    displacements['type'] = 'filelist'
    displacements['data'] = [s1xfile, s1yfile, s1zfile, s2xfile, s2yfile, s2zfile]

    return displacements

class FieldSky:
    '''FieldSky'''
    def __init__(self, **kwargs):

        from . import defaults as fd

        self.ID     = kwargs.get(      'ID',fd.ID)
        self.N      = kwargs.get(       'N',fd.N)
        self.Lbox   = kwargs.get(    'Lbox',fd.Lbox)
        self.Nside  = kwargs.get(   'Nside',fd.Nside)
        self.input  = kwargs.get(   'input',fd.input)
        self.nlpt   = kwargs.get(    'nlpt',fd.nlpt)
        self.gpu    = kwargs.get(     'gpu',fd.gpu)
        self.mpi    = kwargs.get(     'mpi',fd.mpi)
        self.loglev = kwargs.get(  'loglev',fd.loglev)
        self.is64bit= kwargs.get( 'is64bit',fd.is64bit)
        self.peak_per_cell_memory_in_MB = kwargs.get( 'peak_per_cell_memory_in_MB',fd.peak_per_cell_memory_in_MB)
        self.cwsp    = kwargs.get('backend',fd.cwsp)

        self.cube  = kwargs.get(  'cube')
        if self.cube is not None:
            self.input = 'cube'
            self.N    = self.cube.N
            self.Lbox = self.cube.Lbox

        if self.input == 'cube':
            if self.mpi:
                # get MPI info "by hand" for now
                # eventually MPI bookkeeping will be integrated in to xgutil.backend
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                rank  = comm.Get_rank()
                nproc = comm.Get_size()
            else:
                rank  = 0
                nproc = 1
            cube = self.cube
            self.displacements = {}
            self.displacements['type']  = 'arraylist'
            self.displacements['data']  = [cube.s1x.flatten(),
                                           cube.s1y.flatten(),
                                           cube.s1z.flatten(),
                                           cube.s2x.flatten(),
                                           cube.s2y.flatten(),
                                           cube.s2z.flatten()]

            # store the location of local cube data within the global cube here, using the MPI rank information
            # currently set here to y-direction sharding as is the case for sharded exgalsky FFTs, assuming
            # one GPU per MPI rank
            self.displacements['start'] = {}
            self.displacements['stop']  = {}
            self.displacements['start']['x'] = 0
            self.displacements['stop']['x']  = self.N
            self.displacements['start']['y'] = self.N//nproc*rank
            self.displacements['stop']['y']  = self.N//nproc*(rank+1)
            self.displacements['start']['z'] = 0
            self.displacements['stop']['z']  = self.N

    def generate(self, **kwargs):

        from time import time
        times={'t0' : time()}

        import os
        import argparse
        import logging
        from . import libfield as lfm
        from . import defaults as fmd
        import exgaltoolkit.util.ext_interface as ext
        import exgaltoolkit.util.backend as bk

        log = logging.getLogger("LIGHTCONE")

        grid_nside = self.N # cube shape is parameterized by grid_nside; full resolution for websky is 6144
        map_nside  = self.Nside
        L_box      = self.Lbox

        # ------ hardcoded parameters
        comov_lastscatter_Gpc = 13.8  # conformal distance to last scattering surface in Gpc
        zmin                  = 0.05  # minimum redshift for projection (=0.05 for websky products)
        zmax                  = 4.5   # maximum redshift for projection (=4.50 for websky products)

        force_no_mpi          = not self.mpi
        force_no_gpu          = not self.gpu

        kappa_map_filebase = './output/kappa_'+self.ID+f'-{ grid_nside }_nside-{ map_nside }'

        backend = bk.Backend(force_no_mpi=force_no_mpi, force_no_gpu=force_no_gpu,logging_level=-self.loglev)
        backend.print2log(log, f"Backend configuration complete.", level='usky_info')

        if self.cwsp == None:
            backend.print2log(log, f"Computing cosmology...", level='usky_info')
            cosmo_wsp = ext.cosmology(backend, Omega_m=0.31, h=0.68) # for background expansion consistent with websky
            backend.print2log(log, f"Cosmology computed", level='usky_info')
        else:
            cosmo_wsp = self.cwsp

        if self.input == 'lptfiles':
            self.displacements = _get_lpt_displacement_files(backend, grid_nside)

        if self.cwsp == None:
            backend.print2log(log, f"Computing cosmology...", level='usky_info')
            cosmo_wsp = ext.cosmology(backend, Omega_m=0.31, h=0.68) # for background expansion consistent with websky
            backend.print2log(log, f"Cosmology computed", level='usky_info')
        else:
            cosmo_wsp = self.cwsp

            backend.print2log(log, f"Setting up lightcone workspace...", level='usky_info')
        lpt_wsp = lfm.LibField(cosmo_wsp, grid_nside, map_nside, L_box, zmin, zmax)

        backend.print2log(log, f"Computing LPT to kappa map...", level='usky_info')
        kappa_map = lpt_wsp.fieldmap(self.displacements, backend, nlpt=self.nlpt, is64bit=self.is64bit, peak_per_cell_memory_in_MB=self.peak_per_cell_memory_in_MB)
        backend.print2log(log, f"Kappa map computed. Saving to file.", level='usky_info')

        backend.mpi_backend.writemap2file(kappa_map, f"{kappa_map_filebase}_{self.nlpt}LPT.fits")
        backend.print2log(log, f"LIGHTCONE: Kappa map saved. Exiting...", level='usky_info')

        return 0




