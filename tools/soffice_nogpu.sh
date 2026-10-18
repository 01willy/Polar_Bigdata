#!/bin/bash
# soffice_nogpu.sh: LibreOffice 를 GPU 없이 돌린다(OpenCL·GL 끔, CUDA 가시성 0). 사용: bash tools/soffice_nogpu.sh --headless --convert-to pdf --outdir <dir> <file>
export CUDA_VISIBLE_DEVICES="" SAL_DISABLE_OPENCL=1 SAL_DISABLEGL=1 SAL_USE_VCLPLUGIN=svp SAL_NOOPENGL=1 OMP_NUM_THREADS=2
exec nice -n 10 soffice -env:UserInstallation=file:///tmp/claude-1025/lo_nogpu_$$ "$@"
