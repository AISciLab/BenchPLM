# 🚀Getting Started

This guide provides instructions on how to train the model using either **Full Fine-Tuning** or **LoRA (Low-Rank Adaptation)**.

## 1. Installation 📦

First, to prepare the environment to run CARP:

```
$ pip install sequence-models
$ pip install git+https://github.com/microsoft/protein-sequence-models.git
```

Then, install the necessary dependencies for training.

```
$ pip install torch pandas numpy scikit-learn transformers datasets peft
```

------

## 2. Choose a Training Method 🛠️

Select a method based on your computational resources and performance requirements.

### Full Fine-Tuning

This method updates **all** parameters of the model. It generally achieves the highest performance but requires more GPU resources and longer training times.
- **Peptide–protein affinity Script:**

  ```cmd
  python ./full_fine-tuning/fft_regression.py
  ```
  
- **Binary Classification Script:**

  ```
  python ./full_fine-tuning/fft_binary_classification.py --dataset <DATASET_NAME> [OPTIONS]
  ```

- **Multi-class Macro/Micro  Script:**

  ```
  python ./full_fine-tuning/fft_macro_micro.py
  ```

### LoRA (Low-Rank Adaptation) Fine-Tuning

This method freezes most of the pre-trained model's parameters and only trains small, lightweight "adapter" layers. It is much faster and more memory-efficient, making it ideal for rapid experimentation.
- **Peptide–protein affinity Script:**

  ```cmd
  python ./lora/lora_regression.py
  ```
  
- **Binary Classification Script:**

  ```
  python ./lora/lora_binary_classification.py --dataset <DATASET_NAME> [OPTIONS]
  ```

- **Multi-class Macro/Micro Script:**

  ```
  python ./lora/lora_macro_micro.py
  ```

> **Note**: The `<DATASET_NAME>` must be one of `ToxTeller`, `ToxinPred`, or `HemoPI`.

------

## 3. Command-Line Arguments ⚙️

You can customize the training process with the following arguments:

| Parameter      | Description                                     | Default |
| -------------- | ----------------------------------------------- | ------- |
| `--dataset`    | **(Required)** The name of the dataset to use.  | `None`  |
| `--lr`         | The learning rate for training.                 | `2e-5`  |
| `--batch_size` | Batch size for training and evaluation.         | `16`    |
| `--max_epochs` | Maximum number of training epochs.              | `100`   |
| `--patience`   | Number of patience epochs for early stopping.   | `5`     |
| `--num_folds`  | Total number of folds for cross-validation.     | `10`    |
| `--run_time`   | The index of the current fold to run (1-based). | `1`     |

------

## 4. Examples ✨

Here are some specific examples to get you started.

### Example 1: Full Fine-Tuning on the `ToxTeller` Dataset

Run full fine-tuning on the `ToxTeller` dataset using default settings.

```
python ./full_fine-tuning/fft_binary_classification.py --dataset ToxTeller
```

### Example 2: LoRA Tuning with Custom Parameters

Run LoRA tuning on the `HemoPI` dataset with a custom learning rate and batch size, specifying the first run of a 10-fold cross-validation.

```
python ./lora/lora_binary_classification.py \
    --dataset HemoPI \
    --lr 1e-4 \
    --batch_size 8 \
    --num_folds 10 \
    --run_time 1
```
