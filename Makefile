SHELL := /bin/bash

PACKAGE_NAME = maskrcnn_benchmark

error:
	@echo "Empty target is not allowed. Choose one of the targets in the Makefile."
	@exit 2

apt_install:
	sudo apt-get install python3-venv python3-pip

venv:
	python3 -m venv ./venv
	ln -s venv/bin/activate activate
	. ./venv/bin/activate && \
	pip3 install -U pip setuptools wheel

install_pytorch:
	. ./venv/bin/activate && \
	pip install numpy torchvision_nightly && \
	pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html --trusted-host download.pytorch.org

install_maskrcnn:
	. ./venv/bin/activate && \
	pip3 install yacs tqdm && \
	python setup.py build develop

install_pycocotools:
	. ./venv/bin/activate && \
	pip3 install numpy cython && \
	pip3 install matplotlib && \
	pushd ${LOCAL_HOME}/lib/cocoapi/PythonAPI/ && \
	make && \
	popd

install_test:
	. ./venv/bin/activate && \
	pip3 install -U pytest flake8

install_tools:
	. ./venv/bin/activate && \
	pip3 install -U seaborn scikit-image imageio opencv-python

install: venv install_pytorch install_maskrcnn install_pycocotools

dev: install install_test install_tools

install_demo: install
	. ./venv/bin/activate && \
	pip3 install opencv-python
	pip3 install imageio

flake8:
	. ./venv/bin/activate && \
	python -m flake8 --config flake8.config

clean:
	rm -rf `find ${PACKAGE_NAME} -name '*.pyc'`
	rm -rf `find ${PACKAGE_NAME} -name __pycache__`
	rm -rf `find tests -name '*.pyc'`
	rm -rf `find tests -name __pycache__`

clean_all: clean
	rm -rf $(LOCAL_HOME)/lib/cocoapi/PythonAPI/pycocotools/*.so
	rm -rf ${PACKAGE_NAME}/*.so
	rm -rf build/
	rm -rf *.egg-info
	rm -rf venv/
	rm -rf activate
	rm -rf `find . -name '*.log'`
	rm -rf log.txt


# Tools for setting up / syncing remote ubuntu server.

setup:
	scp ${HOME}/projects/scripts/_dircolors ${REMOTE_IP}:~/.dircolors
	scp ${HOME}/projects/scripts/_bash_custom ${REMOTE_IP}:~/.bash_custom
	ssh ${REMOTE_IP} "echo 'source ~/.bash_custom' >> .bashrc"
	ssh ${REMOTE_IP} "mkdir -p /local_storage/projects"
	ssh ${REMOTE_IP} "mkdir -p /local_storage/lib"
	ssh -t ${REMOTE_IP} "cd / && sudo apt update && sudo apt-get install python3-venv python3-dev python-pip"

dry_sync:
	rsync -anv ${PWD} ${REMOTE_IP}:/local_storage/projects/ --delete --exclude=build/ --exclude=venv/ --exclude=activate --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'
	rsync -anv ~/lib/cocoapi ${REMOTE_IP}:/local_storage/lib/ --delete --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'

sync:
	rsync -azP ${PWD} ${REMOTE_IP}:/local_storage/projects/ --delete --exclude=build/ --exclude=venv/ --exclude=activate --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'
	rsync -azP ~/lib/cocoapi ${REMOTE_IP}:/local_storage/lib/ --delete --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'

