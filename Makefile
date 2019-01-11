PACKAGE_NAME = 'maskrcnn-benchmark'

error:
	@echo "Empty target is not allowed. Choose one of the targets in the Makefile."
	@exit 2

apt_install:
	sudo apt-get install python3-venv python3-pip

venv:
	python3 -m venv ./venv
	ln -s venv/bin/activate activate
	. ./venv/bin/activate; \
	pip3 install -U pip setuptools wheel

install_pytorch:
	. ./venv/bin/activate; \
	pip3 install numpy; \
	pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html; \
	pip3 install torchvision

install_maskrcnn:
	. ./venv/bin/activate; \
	pip3 install yacs; \
	python setup.py build develop

install_pycocotools:
	. ./venv/bin/activate; \
	pip3 install numpy cython
	cd ~/external/cocoapi/PythonAPI/ && make && cd ~/external/maskrcnn-benchmark
	# TODO: add pythonpath

install_test:
	. ./venv/bin/activate; \
	pip3 install -U pytest flake8

install_tools:
	. ./venv/bin/activate; \
	pip3 install -U seaborn scikit-image imageio

install: venv install_pytorch install_maskrcnn install_pycocotools install_test

install_demo: install
	. ./venv/bin/activate; \
	pip3 install opencv-python

dev: venv install_package install_test install_tools

test:
	pytest tests -s

ci:
	pytest tests -s

flake8:
	flake8 --ignore=E501,F401,E128,E402,E731,F821 experiment_interface
	flake8 --ignore=E501,F401,E128,E402,E731,F821 tests

clean:
	rm -rf `find maskrcnn_benchmark -name '*.pyc'`
	rm -rf `find maskrcnn_benchmark -name __pycache__`
	rm -rf `find tests -name '*.pyc'`
	rm -rf `find tests -name __pycache__`

clean_all: clean
	rm -rf ../cocoapi/PythonAPI/pycocotools/*.so
	rm -rf maskrcnn_benchmark/*.so
	rm -rf build/
	rm -rf *.egg-info
	rm -rf venv/
	rm -rf activate
	rm -rf `find . -name '*.log'`

.PHONY: venv install_pycocotools install_package install_test install_tools intall dev flake8 clean clean_all test ci dry_sync sync

# Tools for setting up / syncing remote ubuntu server.

setup:
	scp ${HOME}/projects/scripts/_dircolors ${REMOTE_IP}:~/.dircolors
	scp ${HOME}/projects/scripts/_bash_custom ${REMOTE_IP}:~/.bash_custom
	ssh ${REMOTE_IP} "mkdir -p projects"
	ssh ${REMOTE_IP} "sudo apt-get install python3-venv"

dry_sync: clean
	rsync -anv ${PWD} ${REMOTE_IP}:~/projects/ --delete --exclude=venv/ --exclude=activate --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'

sync: clean
	rsync -azP ${PWD} ${REMOTE_IP}:~/projects/ --delete --exclude=venv/ --exclude=activate --exclude-from='${HOME}/projects/scripts/rsync_exclude.txt'


