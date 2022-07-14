# SchedInspector
repo for SchedInspector source code and artifacts
## Installation

### Required Software
* Python 3.7
```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
* OpenMPI 
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

* Virtualenv
```bash
sudo apt install python3.7-dev python3-pip
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3.7 ./venv
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
```

### Clone Deep Batch Scheduler
```bash
git clone https://github.com/DIR-LAB/SchedInspector.git
```

### Install Dependencies
```shell script
cd SchedInspector
pip install -r requirements.txt
```


