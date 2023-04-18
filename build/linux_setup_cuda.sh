curl -O -L https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
export PATH=$PATH:/usr/local/cuda/bin
nvcc --version

bash build/linux_setup.sh
