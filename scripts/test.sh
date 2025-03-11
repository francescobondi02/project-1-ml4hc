#!/bin/bash
#SBATCH -n 24                      # Request 1 core (Python is not MPI-based unless using mpi4py)
#SBATCH --time=00:10:00           # Set a short run-time (10 minutes)
#SBATCH --mem-per-cpu=1000             # Request 1GB memory (Python doesn't need much for Hello World)
#SBATCH -J python_test            # Job name
#SBATCH -o python_test.out        # Output log file
#SBATCH -e python_test.err        # Error log file
#SBATCH --mail-type=END,FAIL      # Notify via email when job ends or fails

module load stack/2024-06 python_cuda/3.9.18 # Load Python module (check available versions on your cluster)
#cd /path/to/script/folder                # Change directory to where your Python script is

python example.py                           # Run the Python script