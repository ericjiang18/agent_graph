# GDesigner

## Overview

We provide the code of our paper. The algorithm implementation code is in `GDesigner` folder, and the experimental code is in `experiments` folder.

## Quick Start

### Install packages

```bash
conda create -n gdesigner python=3.10
conda activate gdesigner
pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" # the BASE_URL of OpenAI LLM backend
API_KEY = "" # for OpenAI LLM backend
```

### Download Datasets

Download MMLU, HumanEval and GSM8K datasets from MMLU, HumanEval and GSM8K. And put them in different folders.

### Run GDesigner on MMLU by running the following scripts

```bash
python experiments/run_mmlu.py --mode FullConnected --batch_size 4 --agent_nums 6 --num_iterations 10 --num_rounds 1 --optimized_spatial
```

The above code verifies the experimental results of the `mmlu` dataset under different topologies.

We also provide experimental code for other datasets and topologies.You can refer to `experiments/run_humaneval.py` and `experiments/run_gsm8k.py`.

For example, if you want to verify the results on the `gsm8k` dataset, you can execute the following command

```bash
python experiments/run_gsm8k.py --mode FullConnected --batch_size 4 --agent_nums 4 --num_iterations 10 --num_rounds 1 --optimized_spatial
```

## Acknowledgement

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
