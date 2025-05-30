import sys
import os
import logging
from .ics import ICs
from .cosmo import CosmologyInterface
from . import defaults as mgd
import exgaltoolkit.util.ext_interface as xgc
import exgaltoolkit.util.log_util as xglogutil
import exgaltoolkit.util.backend  as xgback
import exgaltoolkit.util.jax_util as ju

class Sky:
    '''Sky'''
    def __init__(self, **kwargs):

        self.ID       = kwargs.get(      'ID',mgd.ID)
        self.seed     = kwargs.get(    'seed',mgd.ID)
        self.N        = kwargs.get(       'N',mgd.N)
        self.Niter    = kwargs.get(   'Niter',mgd.Niter)
        self.input    = kwargs.get(   'input',mgd.input)
        self.Lbox     = kwargs.get(    'Lbox',mgd.Lbox)
        self.zInit    = kwargs.get(   'zInit',mgd.zInit)
        self.laststep = kwargs.get('laststep',mgd.laststep)
        self.Nside    = kwargs.get(   'Nside',mgd.Nside)
        self.icw      = kwargs.get(     'icw',mgd.icw)
        self.nlpt     = kwargs.get(    'nlpt',mgd.nlpt)
        self.gpu      = kwargs.get(     'gpu',mgd.gpu)
        self.mpi      = kwargs.get(     'mpi',mgd.mpi)

        from mpi4py import MPI

        self.parallel = False
        self.nproc    = MPI.COMM_WORLD.Get_size()
        self.mpiproc  = MPI.COMM_WORLD.Get_rank()
        self.comm     = MPI.COMM_WORLD
        self.task_tag = "MPI process "+str(self.mpiproc)

        if MPI.COMM_WORLD.Get_size() > 1: self.parallel = True

        # Get cosmo from kwargs or create a default one if not provided
        self.cosmo = kwargs.get('cosmo', CosmologyInterface())
        self.cube = None

    def run(self, **kwargs):
        import jax
        import exgaltoolkit.lpt as lpt
        from time import time
        times={'t0' : time()}

        if not self.parallel:
            self.cube = lpt.Cube(N=self.N,Lbox=self.Lbox,partype=None)
        else:
            ju.distributed_initialize()
            self.cube = lpt.Cube(N=self.N,Lbox=self.Lbox)
        if self.laststep == 'init':
            return 0
        
        err = 0
        seeds = range(self.seed,self.seed+self.Niter)
        i = 0
        for seed in seeds:
            if i==1:
                times={'t0' : time()}
            err += self.generatesky(seed,times)
            i += 1
        xglogutil.summarizetime(None,times,self.comm, self.mpiproc)
        
        return err

    def generatesky(self, seed, times, **kwargs):
        from time import time
        import datetime

        import jax
        import exgaltoolkit.lpt as lpt
        jax.config.update("jax_enable_x64", True)

        import logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if self.mpiproc == 0:
            xglogutil.parprint(f'\nGenerating sky for model "{self.ID}" with seed={seed}')
            
        #### NOISE GENERATION
        self.cube.generate_noise(seed=seed)
        times = xglogutil.profiletime(None, 'noise generation', times, self.comm, self.mpiproc)
        if self.laststep == 'noise':
            return

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        #### NOISE CONVOLUTION TO OBTAIN DELTA
        backend = xgback.Backend(force_no_gpu=True,force_no_mpi=True,logging_level=-logging.ERROR)
        # self.cosmo.get_pspec()
        self.cube.noise2delta(self.cosmo)
        times = xglogutil.profiletime(None, 'noise convolution', times, self.comm, self.mpiproc)
        if self.laststep == 'convolution':
            return

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
                
        #### LPT DISPLACEMENTS FROM DENSITY CONTRAST
        if self.nlpt > 0:
            self.cube.slpt(infield=self.input) # s1 and s2 in cube.[s1x,s1y,s1z,s2x,s2y,s2z]
        times = xglogutil.profiletime(None, 'LPT', times, self.comm, self.mpiproc)
        if self.laststep == 'LPT':
            return

        #### WRITE INITIAL CONDITIONS
        if self.icw:
            self.cosmo.get_growth()
            fname=self.ID+'_'+str(seed)+'_Lbox-'+str(self.Lbox)+'_N-'+str(self.N)+'_proc-'+str(self.mpiproc)
            ics = ICs(self, self.cube, self.cosmo, fname=fname)
            ics.writeics()
            times = xglogutil.profiletime(None, 'write ICs', times, self.comm, self.mpiproc)
        if self.laststep == 'writeics':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        return 0

