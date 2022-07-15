# SchedInspector
repo for SchedInspector source code and artifacts


## Citing

The relevant research paper has been published at HPDC22. If you reference or use SchedInspector in your research, please cite:

```
@inproceedings{zhang2022schedinspector,
	author = {Zhang, Di and Dai, Dong and Xie, Bing},
	title = {SchedInspector: A Batch Job Scheduling Inspector Using Reinforcement Learning},
	year = {2022},
	publisher = {Association for Computing Machinery},
	booktitle = {Proceedings of the 31st International Symposium on High-Performance Parallel and Distributed Computing},
	pages = {97–109}
}
```
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

### Clone SchedInspector
```bash
git clone https://github.com/DIR-LAB/SchedInspector.git
```

### Install Dependencies
```shell script
cd SchedInspector
pip install -r requirements.txt
```


