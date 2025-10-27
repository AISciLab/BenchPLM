# 🧬BenchPLM: Benchmarking protein language models for peptide property and function prediction

## 📜 Description

We conducts a comprehensive benchmark of various protein language models (PLMs) for peptide property and function prediction. The goal is to assess the effectiveness of PLMs on tasks like peptide–protein affinity, toxicity, hemolytic activity, and multi-functional therapeutic peptide prediction. The models are evaluated under two fine-tuning strategies: Full Fine-Tuning and Low-Rank Adaptation (LoRA).

## ⚙️ Benchmarking pipeline

![pipeline](pipeline.png)

## 📊 Databases

We used the following public datasets for benchmarking:

| Dataset                                                      | Tasks            |
| ------------------------------------------------------------ | ---- |
| 🔗 [PPIKB](https://ppikb.duanlab.ac/) | Peptide–protein affinity |
| ☠️ [ToxTeller](https://github.com/comics-asiis/ToxicPeptidePrediction/tree/main) | Peptide toxicity prediction             |
| ☠️ [ToxinPred 3.0](https://webs.iiitd.edu.in/raghava/toxinpred3/) | Peptide toxicity prediction             |
| 🩸 [HemoPI](https://webs.iiitd.edu.in/raghava/hemopi/) |Hemolytic Peptide identification            |
| 💊 [PrMFTP](https://github.com/xialab-ahu/PrMFTP)             | Multi-functional therapeutic peptide prediction |

## 🤖 Benchmarked models
- **ESM2**: Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model[J]. Science, 2023, 379(6637): 1123-1130. [paper](https://www.science.org/doi/10.1126/science.ade2574), [code](https://huggingface.co/facebook/esm2_t36_3B_UR50D).

- **ESM3**: Hayes T, Rao R, Akin H, et al. Simulating 500 million years of evolution with a language model[J]. Science, 2025, 387(6736): 850-858. [paper](https://www.science.org/doi/10.1126/science.ads0018), [code](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1).

- **xTrimoPGLM**: Chen B, Cheng X, Li P, et al. Xtrimopglm: unified 100-billion-parameter pretrained transformer for deciphering the language of proteins[J]. Nature Methods, 2025: 1-12. [paper](https://www.nature.com/articles/s41592-025-02636-z), [code](https://huggingface.co/biomap-research/proteinglm-3b-mlm).

- **ProtT5**: Elnaggar A, Heinzinger M, Dallago C, et al. Prottrans: Toward understanding the language of life through self-supervised learning[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(10): 7112-7127. [paper](https://pubmed.ncbi.nlm.nih.gov/34232869/), [code](https://github.com/agemagician/ProtTrans).

- **AminoBert**: Chowdhury R, Bouatta N, Biswas S, et al. Single-sequence protein structure prediction using a language model and deep learning[J]. Nature Biotechnology, 2022, 40(11): 1617-1623. [paper](https://www.nature.com/articles/s41587-022-01432-w) , [code](https://github.com/zengsihang/AminoBERT-PyTorch).

- **ProtBert**: Elnaggar A, Heinzinger M, Dallago C, et al. Prottrans: Toward understanding the language of life through self-supervised learning[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(10): 7112-7127. [paper](https://pubmed.ncbi.nlm.nih.gov/34232869/),  [code](https://github.com/agemagician/ProtTrans).

- **OmegaPLM**: Wu R, Ding F, Wang R, et al. High-resolution de novo structure prediction from primary sequence[J]. BioRxiv, 2022: 2022.07. 21.500999. [paper](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1), [code](https://github.com/HeliXonProtein/OmegaFold).

- **CARP**: Yang K K, Fusi N, Lu A X. Convolutions are competitive with transformers for protein sequence pretraining[J]. Cell Systems, 2024, 15(3): 286-294. e2. [paper](https://doi.org/10.1016/j.cels.2024.01.008), [code](https://github.com/microsoft/protein-sequence-models).


##  📋 Contents structure

Make sure that your dataset is stored in the following structure so that scripts can read it correctly. 

```
BenchPLM/
├── dataset/            # This catalog is designed to contain the datasets needed for model. Put it in the same directory where the script you want to run is located.
│   ├── HemoPI/
│   ├── PPIKB/
│   ├── PrMFTP/
│   ├── Toxinpred3.0/
│   └── ToxTeller/
│
├── model/              # This directory contains all the methods' full fine-tuning and LoRA code files.
│   ├── AminoBERT/
│   ├── CARP/
│   ├── ESM2/
│   │   ├── README.md   # Specific operation instructions for this model
│   │   ├── full_fine-tuning    # Full fine-tuning script 
│   │   └── lora   # Full fine-tuning script
│   ├── ...             # Other models folder
│   └── xTrimoPGLM/
│
├── data_utils.py  # For peptide–protein affinity data preprocessing, put it in the same directory where the script you want to run is located.
├── evaluation.py  # For multi-tag evaluation, put it in the same directory where the script you want to run is located.
└── README.md
```

## 🚀 Getting Started

For specific details, you can refer to [`./model`](https://github.com/AISciLab/BenchPLM/tree/master/model), where each method includes a file that includes a README file and its implementation on the dataset. For example, the operation step of ESM3 can find in [`./model/ESM3/README.md`](https://github.com/AISciLab/BenchPLM/blob/master/model/ESM3/README.md).
