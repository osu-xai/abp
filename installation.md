ABP Install
============================
Recommended Specs:
 - Python 3.6
 - [Ubuntu 18.04 LTS](https://ubuntu.com/download/desktop)
 - Nvidia CUDA Enabled GPU and comparable CPU

1. Update and install the latest Nvidia GPU drivers for your distribution.
2. Install [Nvidia CUDA](https://developer.nvidia.com/cuda-toolkit)
3. Clone the ABP Repo
    - `git clone https://github.com/osu-xai/abp`
4. Install dependencies
    - `sudo apt install libopenmpi-dev ffmpeg`
5. Install pip packages ([Anaconda recommended](https://www.anaconda.com/distribution/))
    - `pip install -r requirements.txt`
6. Run setup
    - `python setup.py install`
7. Set PYTHONPATH by adding the following to the end of your `.bashrc`, insert correct paths as needed
    - `export PYTHONPATH= <path to where you cloned sc2env>/sc2env:<path to where you cloned abp>/abp`