#!/bin/bash
#SBATCH --job-name=salam2b-full
#SBATCH --account=project_465003647
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=salam2b-full-%j.out
# BH-LLM-Basic 2B puni run — genericki wrapper (CPT ili SFT).
# Upotreba (LUMI):
#   sbatch salamandra-2b-full.sbatch <jobdir> <train_script> <data> <out> <max_steps>
# Primjer CPT-fokus:
#   sbatch salamandra-2b-full.sbatch $J training/train_cpt.py data/cpt_focus.jsonl out/cpt-full 23000
# SAMO UZ GPU ODOBRENJE.
set -euo pipefail
JOBDIR="${1:?jobdir}"; SCRIPT="${2:?train_script}"; DATA="${3:?data}"
OUT="${4:?out}"; MAXSTEPS="${5:?max_steps}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export PYLIBS=/scratch/project_465003647/bh-llm-basic/pylibs
export HF_HUB_CACHE=/scratch/project_465003647/bh-llm-basic/hfcache
export HF_DATASETS_CACHE=/scratch/project_465003647/bh-llm-basic/hfds
export TMPDIR=/scratch/project_465003647/bh-llm-basic/tmp
export TRITON_CACHE_DIR=/scratch/project_465003647/bh-llm-basic/triton
unset TRITON_HIP_ARCH_OVERRIDE
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_PYTHONPATH="$PYLIBS"
mkdir -p "$PYLIBS" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR"
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" pip install -q --target="$PYLIBS" \
  "transformers==5.22.2" "peft==0.21.2" huggingface-hub accelerate 2>&1 | tail -1
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u "/job/${SCRIPT}" --model BSC-LT/salamandra-2b \
    --rev 3f18e03d5e37151681351349b9a42ad17121f4f5 \
    --train "/job/${DATA}" --out "/job/${OUT}" \
    --max-steps "$MAXSTEPS"
echo FULL_DONE
