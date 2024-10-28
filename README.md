# AutoPSV: Automated Process-Supervised Verifier

This repository contains the official implementation of [AutoPSV: Automated Process-Supervised Verifier](https://arxiv.org/abs/2405.16802), accepted at NeurIPS 2024 (poster).

## Code Availability

We will release the code and corresponding finetuned process-enhanced verifier in the near future. Please note that certain portions of the codebase are currently withheld due to confidentiality requirements. We are working to ensure full compliance with open-access requirements before the complete release.

## Framework Overview

![AutoPSV overview](./AUTOCV_main.png)

The AutoPSV framework consists of four key components:

1. **Process-Outcome Verifier**: 
   - Automatically generates process annotations for each reasoning step
   - Monitors confidence variations to evaluate reasoning validity
   - Operates independently of ground truth annotations

2. **Automated Annotation Generation**:
   - Efficiently produces process-level supervision signals
   - Eliminates the need for costly manual annotations
   - Enables scalable training data creation

3. **Large Language Model Training**:
   - Utilizes generated annotations for targeted supervision
   - Enhances model comprehension of reasoning processes
   - Improves output quality through structured learning

4. **Iterative Refinement**:
   - Implements continuous improvement through feedback loops
   - Updates verifier capabilities based on LLM performance
   - Maintains dynamic adaptation to evolving model outputs

## Repository Structure

```
AutoPSV/
├── data_annotation.py
├── train.py
└── utils/
    ├── process_verifier_models.py
    ├── states.py
    └── verifier_datasets.py
```

## Components

### Core Scripts

1. **data_annotation.py**
   - Implements automated process labeling
   - Identifies process calculation hallucinations
   - Streamlines data annotation workflows

2. **train.py**
   - Manages verifier model training
   - Handles data ingestion and model initialization
   - Implements optimization for process- and outcome-supervised verifiers

### Utility Modules

The `utils/` directory contains essential supporting modules:

- **process_verifier_models.py**: Implementation of core verifier architectures
- **states.py**: State management and related functionality
- **verifier_datasets.py**: Dataset handling for model training and validation

## Installation

Install required dependencies using:

```bash
pip install -r requirements.txt
```

## Experimental Results

### Mathematics Benchmarks Performance

| Response Generator | GSM8K Pass@5 | GSM8K Self-Cons. | GSM8K OSV | GSM8K OSV + PSV | MATH Pass@5 | MATH Self-Cons. | MATH OSV | MATH OSV + PSV |
|-------------------|--------------|------------------|------------|-----------------|-------------|-----------------|-----------|----------------|
| Mistral-Instruct  | 69.90        | 50.03            | 61.18      | **61.41**       | 7.7         | 1.64            | 5.10      | **5.30**       |
| Mixtral-Instruct  | 82.30        | 69.06            | 74.91      | **76.04**       | 22.80       | 10.66           | 15.20     | **16.92**      |
| Qwen              | 91.13        | 81.27            | 84.91      | **85.15**       | 56.10       | **40.10**       | 38.94     | 39.36          |

### Commonsense Reasoning Performance

| Response Generator | HellaSwag Pass@5 | HellaSwag Self-Cons. | HellaSwag OSV | HellaSwag OSV + PSV | Winogrande Pass@5 | Winogrande Self-Cons. | Winogrande OSV | Winogrande OSV + PSV | ANLI Pass@5 | ANLI Self-Cons. | ANLI OSV | ANLI OSV + PSV |
|-------------------|------------------|---------------------|---------------|--------------------|--------------------|---------------------|----------------|--------------------| -------------|-----------------|-----------|----------------|
| Mistral-Instruct  | 76.84            | 40.30               | 73.81         | **74.45**          | 91.16              | 58.64                | 79.16          | **79.98**           | 73.4         | 45.6            | 59.8      | **59.3**       |
| Mixtral-Instruct  | 84.05            | 73.67               | 82.83         | **83.62**          | 79.16              | 68.75                | 73.40          | **73.88**           | 68.4         | 59.0            | 62.9      | **64.0**       |
| Qwen-72b          | 95.28            | 85.44               | 93.08         | **93.99**          | 88.63              | 72.21                | **80.34**      | 79.32               | 82.4         | 63.8            | 69.1      | **71.4**       |


## Contributing

We welcome contributions that align with our project goals. Please submit issues or pull requests following our contribution guidelines.

## License

This work is licensed under CC BY 4.0 (Creative Commons Attribution 4.0 International License).

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{lu2024autopsv,
  title={AutoPSV: Automated Process-Supervised Verifier},
  author={Lu, Jianqiao and Dou, Zhiyang and Wang, Hongru and Cao, Zeyu and Dai, Jianbo and Wan, Yingjia and Guo, Zhijiang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
