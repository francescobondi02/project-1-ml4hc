#!/bin/bash
#SBATCH --job-name=embedding_generation
#SBATCH --account=ml4h
#SBATCH --partition=ml4h
#SBATCH --time=5:00:00
#SBATCH --mem=24500
#SBATCH --output=%x-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tiotto@ethz.ch

# 1. Load environment
module load cuda/12.6
source /cluster/courses/ml4h/jupyter/bin/activate

# 2. Start Ollama server in the background

echo "🚀 Starting Ollama server..."
(
    while true; do
        echo "[Ollama] (Re)starting..."
        OLLAMA_MODELS=/cluster/courses/ml4h/llm/models /cluster/courses/ml4h/llm/bin/ollama serve
        echo "[Ollama] Exited. Restarting in 5s..."
        sleep 5
    done
) > ollama_server.log 2>&1 &
OLLAMA_PID=$!

# Wait a few seconds to ensure it's up
sleep 8

# 3. Run your Python script
echo "Generate LLM embeddings..."
python3 project_1/Q4/Q4_2_1_embedding_generation.py
exit_code=$?

# 5. Kill Ollama server if it’s still running
if ps -p $OLLAMA_PID > /dev/null; then
    echo "🛑 Stopping Ollama server..."
    kill $OLLAMA_PID
else
    echo "⚠️ Ollama was not running!"
fi

# 5. Email notification
if [ $exit_code -eq 0 ]; then
    echo "Job $SLURM_JOB_ID completed successfully" | mail -s "[Cluster] Job Success" $SLURM_JOB_MAIL
else
    echo "Job $SLURM_JOB_ID failed with exit code $exit_code" | mail -s "[Cluster] Job Failed" $SLURM_JOB_MAIL
fi