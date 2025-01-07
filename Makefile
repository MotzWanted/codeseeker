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
	wget -nd -r -N -c -np --user $(PHYSIONET_USER) --password $(PHYSIONET_PASS) https://physionet.org/files/phenotype-annotations-mimic/1.20.03/ -P data/meddec/raw || true

.PHONY: prepare-data
prepare-data:
	uv run python src/dataloader/mimiciii/prepare_mimiciii.py data/mimic-iii/raw data/mimic-iii/processed
	uv run python src/dataloader/mimiciv/prepare_mimiciv.py data/mimic-iv/raw data/mimic-iv/processed
	uv run python src/dataloader/mdace/prepare_mdace.py data/mdace/raw data/mdace/processed
	uv run python src/dataloader/meddec/prepare_meddec.py data/meddec/raw data/meddec/processed
	uv run python src/dataloader/snomed/prepare_snomed.py data/snomed/raw data/snomed/processed

.PHONY: install
install:  ## Install the package for development along with pre-commit hooks.
	uv sync

.PHONY: test
test:  ## Run the tests with pytest and generate coverage reports.
	uv run pytest

.PHONY: pre-commit
pre-commit:  ## Run the pre-commit hooks.
	uv run pre-commit run --all-files --verbose

.PHONY: pre-commit-pipeline
pre-commit-pipeline:  ## Run the pre-commit hooks for the pipeline.
	for hook in ${PRE_COMMIT_HOOKS_IN_PIPELINE}; do \
		poetry run pre-commit run $$hook --all-files --verbose; \
	done

.PHONY: clean
clean:  ## Clean up the project directory removing __pycache__, .coverage, and the install stamp file.
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf coverage.xml test-output.xml test-results.xml htmlcov .pytest_cache .ruff_cache
