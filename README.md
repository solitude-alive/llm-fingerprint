# Fingerprint for LLM

This repository is a simple implementation of fingerprint for LLM.

Code for paper [A Fingerprint for Large Language Models](https://arxiv.org/abs/2407.01235)

## Usage

Run `load_model.py` directly.

The format of `config.json` in dir `model` is like:
```json
[
  {
    "name": "gemma",
    "version": "2b",
    "new_version": "2b-1.1-it",
    "standard": "<path of the oracle model>",
    "path": {
      "dir": "<path of dir for fine-tuned model>",
      "sub": "results"
    },
    "dataset": ["samsum", "tatsu-lab"],
    "huggingface": true
  }
]
```

### Requirements
```
torch
transformers
datasets
peft
```
