#!/bin/bash
#SBATCH --job-name=run_llm_predictions
#SBATCH --account=ml4h
#SBATCH --partition=ml4h
#SBATCH --time=05:00:00
#SBATCH --mem=24500
#SBATCH --output=%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tiotto@ethz.ch

# 1. Load CUDA and activate environment
module load cuda/12.6
source /cluster/courses/ml4h/jupyter/bin/activate

# 2. Start Ollama server in background
echo "🚀 Starting Ollama server..."
OLLAMA_MODELS=/cluster/courses/ml4h/llm/models /cluster/courses/ml4h/llm/bin/ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Wait briefly to ensure Ollama is ready
sleep 8

# 3. Run your Python evaluation script
echo "📊 Running LLM evaluation..."
python3 project_1/Q4/Q4_1_2_llm_prediction.py
exit_code=$?

# 5. Kill Ollama server if it’s still running
if ps -p $OLLAMA_PID > /dev/null; then
    echo "🛑 Stopping Ollama server..."
    kill $OLLAMA_PID
else
    echo "⚠️ Ollama was not running!"
fi

# 5. Notify on job result
if [ $exit_code -eq 0 ]; then
    echo "Job $SLURM_JOB_ID completed successfully" | mail -s "[Cluster] LLM Eval Success" $SLURM_JOB_MAIL
else
    echo "Job $SLURM_JOB_ID failed with exit code $exit_code" | mail -s "[Cluster] LLM Eval Failed" $SLURM_JOB_MAIL
fi