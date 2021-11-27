CONDA_ENV_NAME:=stabilize2p

ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
	ENV_DIR=$(shell conda info --base)
	MY_ENV_DIR=$(ENV_DIR)/envs/$(CONDA_ENV_NAME)
	CONDA_ACTIVATE=. $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
endif

.PHONY: github_install remove-env install_environment.yml install-env reinstall-env update-env

%.github_install: path=$(@:%.github_install=%)
%.github_install: name=$(word 2,$(subst /, ,$(path)))
%.github_install:
	# $(CONDA_ACTIVATE) $(CONDA_ENV_NAME); pip install -e git+"https://github.com/$(path)#egg=$(name)"
	pip install -e git+"https://github.com/$(path)#egg=$(name)"
github_install: NeLy-EPFL/ofco.github_install NeLy-EPFL/utils2p.github_install NeLy-EPFL/utils_video.github_install adalca/pystrum.github_install adalca/neurite.github_install voxelmorph/voxelmorph.github_install

remove-env:
	conda env remove -n $(CONDA_ENV_NAME)

install_environment.yml:
	mkdir -p models
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","") # check if the directory is there
	@echo ">>> Found $(CONDA_ENV_NAME) environment in $(MY_ENV_DIR). Skipping installation..."
else
	@echo ">>> Detected conda, but $(CONDA_ENV_NAME) is missing in $(ENV_DIR). Installing ..."
	conda env create -f environment.yml
endif
else
	@echo ">>> Install conda first."
	exit
endif

install-env: install_environment.yml github_install

reinstall-env:
	remove-env install-env
update-env:
	conda env update -f environment.yml --prune
