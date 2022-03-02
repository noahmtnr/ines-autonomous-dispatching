# ines-autonomous-dispatching
InES Team Project SS22 - Autonomous Dispatching System

<b>To create conda environment based on spec file</b><br/>
$ conda env create -f win_env.yaml <br/>
OR <br>
$ conda env create -f mac_env.yaml <br/><br/>

<b/>To activate conda environment, use</b><br/>
$ conda activate ines-ad

## Package Installation process:
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
  - $ pip install gym
  - $ pip install pygame
6) Install RLlib within teamproject env
  - $ pip install -U ray
  - $ pip install ray[rllib]
