curl -O -L https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

nvcc --version

pip install toml
python build/add_cuda_to_version.py

bash build/linux_setup.sh
