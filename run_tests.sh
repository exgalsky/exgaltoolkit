#!/bin/bash

cd /global/cfs/cdirs/mp107d/exgal/users/malvarez/exgaltoolkit

source ./loadenv.sh

python -m pytest tests/ -v -s
