# Inspired by: https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# 			   https://www.thapaliya.com/en/writings/well-documented-makefiles/

.DEFAULT_GOAL := help

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: download-data
download-data :
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/snomed-ct-entity-challenge/1.0.0/ -P data/snomed/raw || true
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/mimiciv/2.2/ -P data/mimic-iv/raw || true
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/mimic-iv-note/2.2/ -P data/mimic-iv-note/raw || true
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/mimiciii/1.4/ -P data/mimic-iii/raw || true
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/meddec/1.0.0/ -P data/meddec/raw || true

.PHONY: prepare-data
prepare-data:
	poetry run python src/dataloader/mimic-iii/pipelines/prepare_mimiciii.py data/mimic-iii/raw data/mimic-iii/processed
	poetry run python src/dataloader/mimic-iv/pipelines/prepare_mimiciv.py data/mimic-iv/raw data/mimic-iv/processed
	poetry run python src/dataloader/mdace/pipelines/prepare_mdace.py data/mdace/raw data/mdace/processed

.PHONY: install
install:  ## Install the package for development along with pre-commit hooks.
	poetry install --with dev --with test
	poetry run pre-commit install

.PHONY: test
test:  ## Run the tests with pytest and generate coverage reports.
	poetry run pytest -vvs tests --typeguard-packages=src --junitxml=test-results.xml --cov --cov-report=xml \
		--cov-report=html --cov-report=term

.PHONY: pre-commit
pre-commit:  ## Run the pre-commit hooks.
	poetry run pre-commit run --all-files --verbose

.PHONY: pre-commit-pipeline
pre-commit-pipeline:  ## Run the pre-commit hooks for the pipeline.
	for hook in ${PRE_COMMIT_HOOKS_IN_PIPELINE}; do \
		poetry run pre-commit run $$hook --all-files --verbose; \
	done

.PHONY: clean
clean:  ## Clean up the project directory removing __pycache__, .coverage, and the install stamp file.
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf coverage.xml test-output.xml test-results.xml htmlcov .pytest_cache .ruff_cache
