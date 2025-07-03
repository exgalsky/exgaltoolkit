#!/bin/bash

cd /global/cfs/cdirs/mp107d/exgal/users/malvarez/exgaltoolkit

source ./loadenv.sh

# Check if running in serial mode
if [ "$1" = "--serial" ]; then
    # Regular serial execution
    echo "Running example in serial mode..."
    python ./examples/minimal_example_serial_newapi.py |& tee examplelog 
else
   srun \
    -n 4\
    --qos=interactive \
    -N 1 \
    --time=10 \
    -C gpu \
    -A cosmosim \
    --gpus-per-node=4 \
    --exclusive \
    python ./examples/minimal_example_serial_newapi.py |& tee examplelog
fi

