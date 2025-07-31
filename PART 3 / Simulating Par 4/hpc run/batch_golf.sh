#!/bin/bash
#SBATCH --job-name=par4sim
#SBATCH --output=logs/par4sim_%j.out
#SBATCH --error=logs/par4sim_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=116
#SBATCH --time=24:00:00
#SBATCH --mem=64G

module load python/3.10 

echo "Job started on $(date)"
python mainsimulationrunner.py
echo "Job ended on $(date)"
