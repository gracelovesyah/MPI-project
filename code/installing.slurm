#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00

module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong

mkdir ~/virtualenv
virtualenv ~/virtualenv/python3.7.4
source ~/virtualenv/python3.7.4/bin/activate
pip install pandas fastparquet ijson pyarrow

srun -n 3 python3 ./script/MPI_Series_1.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s