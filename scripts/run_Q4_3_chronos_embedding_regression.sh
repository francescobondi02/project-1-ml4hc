#!/bin/bash
#SBATCH --job-name=chronos_embedding_regression
#SBATCH --account=ml4h
#SBATCH --partition=ml4h
#SBATCH --time=03:00:00
#SBATCH --mem=24000
#SBATCH --output=%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tiotto@ethz.ch

# 1. Load CUDA and activate environment
module load cuda/12.6
source .venv/bin/activate

# 2. Run the Chronos embedding script
echo "🚀 Starting Chronos embedding job..."
python3 project_1/Q4/Q4_3_chronos_embedding_regression.py
exit_code=$?

# 3. Notify on job result
if [ $exit_code -eq 0 ]; then
    echo "Job $SLURM_JOB_ID completed successfully" | mail -s "[Cluster] Chronos Embed Success" $SLURM_JOB_MAIL
else
    echo "Job $SLURM_JOB_ID failed with exit code $exit_code" | mail -s "[Cluster] Chronos Embed Failed" $SLURM_JOB_MAIL
fi