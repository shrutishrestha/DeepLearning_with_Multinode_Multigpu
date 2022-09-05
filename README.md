# DeepLearning_with_Multinode_Multigpu

for running the file, 
srun -p qGPU24 -A AAS3S2 --nodes=1 --ntasks-per-node=1 --gres=gpu:V100:2 --time=03:00:00 --mem=8G --pty -w acidsgcn001 bash
