# ines-autonomous-dispatching
InES Team Project SS22 - Autonomous Dispatching System

## Package Installation process:
Donwload Anaconda: https://www.anaconda.com/products/individual
1) Install Python version 3.9 in anaconda prompt
  - $ conda install python = 3.9
3) create new environment
  - open anaconda prompt
  - $ conda config --prepend channels conda-forge
  - $ conda create -n teamproject --strict-channel-priority osmnx
  - $ conda activate teamproject
  - open anaconda navigator, select environment teamproject in the drop down menu and install Jupyter Notebook
4) Install pytorch within teamproject environment
  - $ pip3 install torch torchvision torchaudio
5) Install open AI gym
 - $ pip install pygame
 - $ pip install gym
6) Install RLlib within teamproject env
  - $ pip install -U ray
  - $ pip install ray[rllib]
