.PHONY: install

%.github_install: path=$(@:%.github_install=%)
%.github_install: name=$(word 2,$(subst /, ,$(path)))
%.github_install:
	pip install -e git+"https://github.com/$(path)#egg=$(name)"
install: NeLy-EPFL/ofco.github_install NeLy-EPFL/utils2p.github_install NeLy-EPFL/utils_video.github_install adalca/pystrum.github_install adalca/neurite.github_install voxelmorph/voxelmorph.github_install
