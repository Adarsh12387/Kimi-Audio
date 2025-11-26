# --- start script ---
conda activate nkimi

# 1) Basic sanity print (optional)
python - <<'PY'
import torch, sys
print("Python:", sys.executable)
print("Torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
PY

# 2) Uninstall any existing flash-attn
pip uninstall -y flash-attn flash_attn || true

# 3) Ensure build deps available
pip install --upgrade pip setuptools wheel cmake ninja psutil

# 4) Ensure CUDA env (adjust if CUDA installed elsewhere)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 5) Set compiler to a GCC >= 11 if available (adjust path if needed)
# If you have module system, prefer 'module load gcc/11 cuda/12.4' instead of setting CC/CXX
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# 6) Build flags: target H100 (sm_90). Change if you have other GPUs.
export TORCH_CUDA_ARCH_LIST="9.0"
# Try ABI = 1 (match most recent PyTorch builds). If it fails try 0.
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"

# 7) Clone flash-attention and build in editable mode (clean build)
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention-build
cd /tmp/flash-attention-build

# Clean any previous build artifacts (safe)
python -m pip install --upgrade pip
python setup.py clean || true
rm -rf build dist *.egg-info || true

# Build & install using your current env's torch (no build isolation)
python -m pip install -e . --no-build-isolation

# 8) Verify import
python - <<'PY'
import importlib, sys
try:
    import flash_attn
    print("flash_attn version:", flash_attn.__version__)
    print("OK: flash_attn import succeeded")
except Exception as e:
    print("ERROR importing flash_attn:", e)
    sys.exit(1)
PY

# --- end script ---

