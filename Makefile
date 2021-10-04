%.github_install: path=$(@:%.github_install=%)
%.github_install: name=$(word 2,$(subst /, ,$(path)))
%.github_install:
	pip install -e git+"https://github.com/$(path)#egg=$(name)"
github_install: NeLy-EPFL/ofco.github_install NeLy-EPFL/utils2p.github_install NeLy-EPFL/utils_video.github_install

remove-env:
	conda env remove -n 2p-stabilizer
install-env: github_install
	conda env create -f environment.yml
reinstall-env:
	remove-env install-env
update-env:
	conda env update -f environment.yml --prune
