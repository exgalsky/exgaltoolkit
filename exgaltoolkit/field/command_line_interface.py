def main():

    # for full res websky 1LPT at nside=1024:
    #   export LPT_DISPLACEMENTS_PATH=/pscratch/sd/m/malvarez/websky-displacements/
    #   srun -n $nproc --gpus-per-task=1 xgfield --N 6144 --Nside 1024 --input lptfiles

    import exgaltoolkit.field
    from . import defaults as fmd

    import sys
    import argparse
    from . import fieldsky

    parser = argparse.ArgumentParser(description='Commandline interface to fieldmock')

    parser.add_argument('ID',                         help=f'model ID [{fmd.ID}]',          type=str)
    parser.add_argument('--N',     default=fmd.N,     help=f'grid dimension [{fmd.N}]',     type=int)
    parser.add_argument('--Lbox',  default=fmd.Lbox,  help=f'box size in Mpc [{fmd.Lbox}]', type=int)
    parser.add_argument('--Nside', default=fmd.Nside, help=f'healpix Nside [{fmd.Nside}]',  type=int)
    parser.add_argument('--input', default=fmd.input, help=f'field input [{fmd.input}]',    type=str)
    parser.add_argument('--gpu',   default=fmd.gpu,   help=f'use GPU [{fmd.gpu}]', action=argparse.BooleanOptionalAction)
    parser.add_argument('--mpi',   default=fmd.mpi,   help=f'use MPI [{fmd.mpi}]', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    sky = fieldsky.FieldSky(ID    = args.ID,
                            N     = args.N,
                            Lbox  = args.Lbox,
                            Nside = args.Nside,
                            input = args.input,
                            gpu   = args.gpu,
                            mpi   = args.mpi)

    return sky.generate()

if __name__ == "__main__":
    import sys
    sys.exit(main())