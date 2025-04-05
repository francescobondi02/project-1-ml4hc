#!/bin/bash
#SBATCH -n 24                      # Request 1 core (Python is not MPI-based unless using mpi4py)
#SBATCH --time=00:30:00           # Set a short run-time (10 minutes)
#SBATCH --mem-per-cpu=1000             # Request 1GB memory (Python doesn't need much for Hello World)
#SBATCH -J tokenizer            # Job name
#SBATCH -o tokenizer.out        # Output log file
#SBATCH -e tokenizer.err        # Error log file
#SBATCH --mail-type=END,FAIL      # Notify via email when job ends or fails

module load stack/2024-06 python_cuda/3.9.18 # Load Python module (check available versions on your cluster)
#cd /path/to/script/folder                # Change directory to where your Python script is
export PYTHONPATH=${PYTHONPATH}:$(pwd)

python project_1/Q2/Q2_3_tokenizing_TZV.py                        # Run the Python script