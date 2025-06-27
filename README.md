# Agent-Diffusion: Guided Topology Diffusion for Multi-Agent Systems

## Overview

This repository contains the code for a multi-agent framework designed to solve complex tasks. The core of this project is **Guided Topology Diffusion (GTD)**, a novel methodology for dynamically generating optimal communication structures for teams of AI agents.

The GTD framework consists of three main components:
1.  A **Generator** (`ConditionalDiscreteGraphDiffusion`): A diffusion model trained to generate agent graph topologies.
2.  A **Guider** (`GuidedGeneration`): A module that uses a reward model to steer the generation process towards high-performing graphs.
3.  A **Proxy Reward Model**: A model trained to predict the performance of a given graph topology on a task, which is used to guide the generator.

The main experimental script is `experiments/run_gsm8k.py`, which demonstrates the GTD pipeline on the GSM8K benchmark.

## Quick Start

### 1. Installation

Set up the environment and install the required packages.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file from the template and add your API keys.

```bash
cp template.env .env
```

Then, edit `.env` with your credentials:
```
BASE_URL = "YOUR_BASE_URL" # The BASE_URL of your LLM backend
API_KEY = "YOUR_API_KEY"   # The API_KEY for your LLM backend
```

### 3. Download Datasets

Download the GSM8K dataset and place it in the `datasets/gsm8k/` directory. The main script expects to find `gsm8k.jsonl` there.

### 4. Running the GTD Pipeline for GSM8K

The entire GTD pipeline can be executed using the `run_gsm8k.sh` script. This script automates the three critical phases of the GTD methodology.

```bash
# Make sure the script is executable
chmod +x agent_diffusion/run_gsm8k.sh

# Run the script from the root of the project
./agent_diffusion/run_gsm8k.sh
```

The script will execute the following phases in order:

*   **Phase 1: Dataset Generation**: It first runs experiments with baseline, fixed topologies (e.g., fully-connected, chain) to generate a dataset of graph-performance pairs. This dataset is saved as `gtd_gsm8k_dataset.jsonl`.
*   **Phase 2: Model Training**: It then uses the generated dataset to train both the Proxy Reward Model and the Diffusion Model. The trained models are saved as `proxy_model_gsm8k.pth` and `diffusion_model_gsm8k.pth`.
*   **Phase 3: Inference**: Finally, it runs the main experiment using the trained GTD framework. For each task, the framework generates a bespoke communication graph, and the agents collaborate using that graph to solve the problem.

You can also run each phase manually by uncommenting the desired section in `run_gsm8k.sh`.
