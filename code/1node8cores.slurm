#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-12:00:00

module load python/3.7.4
module load mpi4py/3.0.2-timed-pingpong
source ~/virtualenv/python3.7.4/bin/activate

mpiexec -n 8 python ./script/MPI_Series_1.py
mpiexec -n 8 python ./script/MPI_Series_2.py
mpiexec -n 8 python ./script/MPI_Series_3_q1.py
mpiexec -n 8 python ./script/MPI_Series_3_q2.py
mpiexec -n 8 python ./script/MPI_Series_3_q3.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s