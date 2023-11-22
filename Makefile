#  Write set of makefile to make your life easier
guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Required variable $* not set"; \
		exit 1; \
	fi

venv: guard-VENV_NAME
	python3 -m venv ${VENV_NAME}

install:
	python3 -m pip install --upgrade pip setuptools
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

freeze_pip:
	python3 -m pip list --format=freeze > requirements.txt