#!/bin/bash
#SBATCH --job-name=cycle_translate
#SBATCH --output=translation_logs/translate_%j.out
#SBATCH --error=translation_logs/translate_%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist alpha2

echo "=============================="
echo "Starting SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================="

# Activate environment
source /opt/conda/bin/activate dpm-pc-gen


# Optional: Print GPU info
nvidia-smi

# Naming convention:
# sim = yesSC (WITH SPACE CHARGE)
# exp = noSC (WITHOUT SPACE CHARGE)

# Run translation script
python mass_translate.py \
    --sim_model ../logs_gen/fission_sim_yesSC/ckpt_82000.000000_3854047.pt \
    --exp_model ../logs_gen/fission_sim_noSC/ckpt_87500.000000_3675042.pt \
    --sim_data ../data/fission_data/Fission_sim_yesSC_sampled_XYZC_filtered_scaled.npy \
    --exp_data ../data/fission_data/Fission_sim_noSC_sampled_XYZC_filtered_scaled.npy \
    --trans_ratio 1

echo "=============================="
echo "Job Finished"
echo "=============================="
