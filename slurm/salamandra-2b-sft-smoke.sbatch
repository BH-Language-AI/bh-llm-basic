#!/bin/bash
#SBATCH --job-name=salam2b-sft-smoke
#SBATCH --account=project_465003647
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=salam2b-sft-smoke-%j.out
# BH-LLM-Basic 2B SFT smoke — Salamandra-2B LoRA bf16 na 1xMI2500 (ROCm, bez bitsandbytes).
# Recept: provjereni bh-vision-smo2b predlozak (isti projekt/particija/kontejner/kese).
# Upotreba (LUMI login node, nakon sto job-dir stigne preko Oracle skocnog cvora):
#   sbatch salamandra-2b-sft-smoke.sbatch <job-dir> [max_steps]
# job-dir sadrzi: training/train_sft.py, data/smoke_sft.jsonl (renderirano
# scripts/render_salamandra.py lokalno). Izlazi: out/adapter-smoke/.
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch salamandra-2b-sft-smoke.sbatch <job-dir> [max_steps]}"
MAXSTEPS="${2:-100}"
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
  "transformers==5.22.2" "peft==0.21.2" datasets huggingface-hub accelerate 2>&1 | tail -1
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for g in ('torch*','nvidia*','triton*','cuda*','torchaudio*','torchvision*') for p in glob.glob(os.path.join('$PYLIBS', g))]; print('PYLIBS-CIST')"
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/training/train_sft.py --model BSC-LT/salamandra-2b \
    --rev 1812f3829e123d08541b3ea2fede5366f29d3ee92 \
    --train /job/data/smoke_sft.jsonl --out /job/out/adapter-smoke \
    --max-steps "$MAXSTEPS"
echo SMOKE_DONE
