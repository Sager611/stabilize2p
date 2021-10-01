remove-env:
	conda env remove -n 2p-stabilizer
install-env:
	conda env create -f environment.yml
reinstall-env:
	remove-env install-env
update-env:
	conda env update -f environment.yml --prune
