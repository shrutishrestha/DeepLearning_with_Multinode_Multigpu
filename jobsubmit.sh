#!/bin/bash
#SBATCH --job-name=pytorch_multinode
#SBATCH -w acidsgcn001,acidsgcn002
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --partition=qGPU48
#SBATCH --mem=128gb
#SBATCH --gres=gpu:V100:2
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=sshrestha8@student.gsu.edu
#SBATCH --account=csc344r73
#SBATCH --output=outputs/output_%j
#SBATCH --error=errors/error_%j

cd /scratch
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID

iget -r /arctic/projects/csc344s73/nsightdemo/arctic_run_folders

source /userapp/virtualenv/SR_ENV/venv/bin/activate
export NCCL_DEBUG=INFO
srun python -u arctic_run_folders/basecode_multinode_multigpu.py

cd /scratch
icd /arctic/projects/csc344s73
iput -rf $SLURM_JOB_ID

