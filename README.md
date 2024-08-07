# CARC Workshop on Running Deep Learning Applications

### About
The material in this repo contains workshop materials related to running deep learning applications in HPC system. 

### Software Environment Setup
First login to CARC OnDemand: https://ondemand.carc.usc.edu/ and request a discovery shell terminal within OpenOnDemand. 

We will use Conda to build software packages. If it is the first time you are using Conda, make sure you follow the guide of how to use Conda with this link: https://www.carc.usc.edu/user-guides/data-science/building-conda-environment
```bash
$ salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=1:00:00
$ mamba create --name torch-env
$ mamba activate torch-env
$ mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ mamba install line_profiler --channel conda-forge   #optional, needed if you want to use line_profiler function within your code
```

### Clone this repo and start learning how to run deep learning applications in HPC system. 
$ git clone https://github.com/uschpc/Running-DL-Applications.git
$ cd Running-DL-Applications
