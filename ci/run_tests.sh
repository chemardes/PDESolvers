#!/bin/bash
# Make sure the tests will FAIL if it has to
set -euxo pipefail

export PYTHONPATH=$(pwd)

## PYTHON TESTS
## ------------------------------------------
## Set up your Python environment
pip install virtualenv
virtualenv -p python3.10 venv
source venv/bin/activate

pip install -r requirements.txt

pytest --cov=pdesolvers pdesolvers/tests/

# CUDA TESTS
# ------------------------------------------
cd GPUSolver

cpp_version=17 # default
sm_arch=86

cmake -DCPPVERSION=${cpp_version} -DSM_ARCH=${sm_arch} -S . -B ./build -Wno-dev
cmake --build ./build
ctest --test-dir ./build/tests --output-on-failure

cd ./build
mem=$(/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check=full ./GPU_runner)
grep "0 errors" <<< "$mem"
