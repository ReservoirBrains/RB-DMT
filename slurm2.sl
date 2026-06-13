#!/bin/bash
#SBATCH --job-name=owt
#SBATCH --output=output_ohbm_%j.txt
#SBATCH --error=error_ohbm_%j.txt

# Arrêter le script immédiatement si une commande échoue
set -e

# Initialize env
eval "$(conda shell.bash hook)"
conda init
conda activate /beegfs/ybendiou/ohbm/ohbm_venv/

cd /beegfs/ybendiou/ohbm/


srun python train1.py --out_temp r3_temp.csv --out_final r3_final.csv --batch_size 8 --grad_accum 4 --device cuda:0 --m_units 16 --m_dim 128 --a_dim 64 & 
srun python train2.py --out_temp r4_temp.csv --out_final r4_final.csv --batch_size 8 --grad_accum 4 --device cuda:1 --m_units 16 --m_dim 128 --a_dim 64 & 

wait