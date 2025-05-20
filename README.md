# Code Like Humans (CLH): A Multi-Agent Solution for Medical Coding

This repository contains experiments for LLM-based medical code assignment using a multi-agent approach.

## Prerequisites

Before running the experiments, you'll need:

1. A PhysioNet account with access to MIMIC-III, MIMIC-IV, and other medical datasets
   - Register at [PhysioNet](https://physionet.org/)
   - Request access to the required datasets:
     - MIMIC-III Clinical Database
     - MIMIC-IV Clinical Database
     - MIMIC-IV Notes
     - SNOMED-CT Entity Challenge Dataset
     - MedDEC Dataset
     - Phenotype Annotations MIMIC Dataset

2. Python environment management tools:
   - [UV](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
     ```bash
     # Install UV (macOS/Linux)
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

## Data Setup

### 1. Download the Required Datasets

The repository includes a Makefile that automates the download of all required datasets. To download the data, run:

```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password
make download-data
```

This command will download all necessary datasets to their respective directories under the `data/` folder:
- MIMIC-III → `data/mimic-iii/raw/`
- MIMIC-IV → `data/mimic-iv/raw/`
- MIMIC-IV Notes → `data/mimic-iv-note/raw/`

### 2. Prepare the Data

After downloading the raw data, you need to process it into the format used by the experiments:

```bash
make prepare-data
```

This command will:
- Process MIMIC-III data
- Process MIMIC-IV data
- Prepare MDACE dataset

## Installation

To install all required dependencies:

### CPU-only Installation
```bash
uv sync
```

### GPU-enabled Installation
If you plan to use GPU acceleration, install with GPU extras:
```bash
uv sync --extra GPU
```

This will:
- Install the package in development mode
- Set up pre-commit hooks
- Install all required dependencies using UV
- Include GPU-specific packages if GPU extras are enabled

## Running Experiments

The repository contains several experiment scripts that can be run individually or as an end-to-end pipeline.

### Individual Agent Experiments

Each agent in the multi-agent pipeline has its own experiment script in the `experiments/` folder:

```bash
# Run Analysis Agent experiments
uv run python experiments/1_analyse_agent.py

# Run Location Agent experiments
uv run python experiments/2_locate_agent.py

# Run Verification Agent experiments
uv run python experiments/3_verify_agent.py

# Run Assignment Agent experiments
uv run python experiments/4_assign_agent.py
```

### End-to-End Benchmark

To run the complete multi-agent pipeline benchmark:

```bash
uv run python experiments/benchmark.py
```

This script orchestrates all agents (analyze, locate, verify, and assign) to perform the complete medical code assignment task.

### Model Fine-tuning

For fine-tuning the models:

```bash
# Standard fine-tuning
uv run python experiments/run_training.py

# Fine-tuning with GPRO optimization
uv run python experiments/run_gpro_unsloth.py

# Fine-tuning with Unsloth optimization
uv run python experiments/run_unsloth.py
```

Each script can be configured through command-line arguments. Use the `--help` flag to see available options:

```bash
uv run python experiments/benchmark.py --help
```

## Ethical Considerations and Limitations

CLH is an open-source research project designed to explore the potential of large language models in medical coding automation. While we strive for transparency and accessibility by releasing our code under the MIT license, there are important ethical considerations and limitations to be aware of:

### Research Use Only
This codebase is intended **strictly for research purposes**. It should not be used in clinical practice or for making medical coding decisions that affect patient care, billing, or healthcare operations. The models and approaches demonstrated here have not undergone the rigorous validation required for clinical deployment.

### Limitations
- The system's outputs cannot be guaranteed to be accurate or complete
- Models may exhibit biases present in the training data
- Performance can vary significantly across different medical specialties and coding systems
- The system has not been validated against current medical coding standards and guidelines
- No warranty is provided for the accuracy or reliability of the code assignments

### Responsible Development
Before considering any adaptation of this work for production use:
- Extensive validation must be performed
- Clinical expertise must be incorporated
- Compliance with relevant healthcare regulations must be ensured
- Proper safety measures and human oversight must be implemented

We encourage open collaboration and research in this area while emphasizing the critical importance of responsible development practices in healthcare applications. 
