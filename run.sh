#!/bin/bash -l
#SBATCH --job-name=portfolio_opt
#SBATCH --output=portfolio_opt_%A_%a.out
#SBATCH --error=portfolio_opt_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=26G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-19

echo "Running job array ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Starting at $(date)"

# Create a virtual environment specific to this job array (shared across tasks)
VENV_DIR="$HOME/venvs/portfolio_opt_${SLURM_ARRAY_JOB_ID}"

# Only create the environment for the first task (task 0)
# Other tasks will wait until the environment is ready
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install numpy scipy yfinance pycma pennylane PyPortfolioOpt
    # Signal that the environment is ready
    touch "${VENV_DIR}/env_ready"
else
    # Wait for the environment to be ready
    echo "Waiting for environment to be created by task 0..."
    while [ ! -f "${VENV_DIR}/env_ready" ]; do
        sleep 5
    done
    echo "Environment is ready, activating..."
    source $VENV_DIR/bin/activate
fi

# Set total number of batches
TOTAL_BATCHES=20

# Run the script
python experiments.py ${SLURM_ARRAY_TASK_ID} ${TOTAL_BATCHES}

# Clean up only if this is the last task
#if [ "${SLURM_ARRAY_TASK_ID}" -eq 19 ]; then
#    echo "Cleaning up environment"
#    rm -rf $VENV_DIR
#fi

echo "Job finished at $(date)"