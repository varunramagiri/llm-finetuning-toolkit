# 🧬 LLM Fine-Tuning Toolkit

> **Parameter-efficient fine-tuning (PEFT) for GPT, BERT, T5, LLaMA & Mistral** — LoRA · QLoRA · MLflow · SageMaker · Vertex AI · One-command deployment

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E?style=flat-square)](https://huggingface.co/transformers)
[![MLflow](https://img.shields.io/badge/MLflow-2.12+-0194E2?style=flat-square)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 🎯 Overview

Production-ready toolkit for fine-tuning large language models on domain-specific enterprise datasets using **LoRA (Low-Rank Adaptation)** and **QLoRA (4-bit quantized LoRA)** — dramatically reducing GPU memory requirements without sacrificing model quality.

Built from fine-tuning work across **insurance policy summarization** (Nationwide), **clinical NLP** (CVS Health), and **document classification** tasks. Includes training, evaluation, experiment tracking, and cloud deployment — all scriptable via CLI or Python API.

**Supported tasks:** Summarization · Classification · QA · NER · Instruction following · Chat fine-tuning

**Supported models:** GPT-2 · BERT · RoBERTa · T5 · FLAN-T5 · LLaMA 2/3 · Mistral 7B · Falcon

---

## 📁 Folder Structure

```
llm-finetuning-toolkit/
├── src/
│   ├── training/
│   │   ├── trainer.py              # Core PEFT trainer class
│   │   ├── lora_config.py          # LoRA: rank, alpha, target modules
│   │   ├── qlora_config.py         # QLoRA: 4-bit NF4, double quant
│   │   └── callbacks.py            # Early stopping, checkpoint saving
│   ├── data/
│   │   ├── dataset_loader.py       # HuggingFace Hub + custom JSONL
│   │   ├── preprocessor.py         # Tokenization, padding, truncation
│   │   ├── augmentation.py         # Back-translation, synonym swap
│   │   └── quality_filter.py       # Dedupe, length filter, toxicity
│   ├── evaluation/
│   │   ├── metrics.py              # BLEU, ROUGE, F1, BERTScore
│   │   ├── bias_detector.py        # Fairness metrics across demographics
│   │   ├── safety_eval.py          # ToxiGen, BBQ safety benchmarks
│   │   └── benchmark.py            # Throughput, latency, memory profiling
│   ├── inference/
│   │   ├── predictor.py            # Single inference with adapter loading
│   │   ├── batch_predictor.py      # Async batch inference (vLLM)
│   │   └── quantizer.py            # Post-training quant (ONNX / TensorRT)
│   └── deployment/
│       ├── sagemaker_deploy.py     # AWS SageMaker real-time endpoint
│       ├── vertex_deploy.py        # GCP Vertex AI endpoint
│       ├── azure_deploy.py         # Azure ML managed endpoint
│       └── vllm_server.py          # Local vLLM OpenAI-compatible server
├── configs/
│   ├── lora_bert_classification.yaml
│   ├── qlora_llama2_7b_instruct.yaml
│   ├── lora_t5_summarization.yaml
│   └── qlora_mistral_7b_chat.yaml
├── scripts/
│   ├── train.py                    # CLI: python scripts/train.py --config ...
│   ├── evaluate.py                 # CLI: python scripts/evaluate.py ...
│   ├── merge_adapters.py           # Merge LoRA weights → base model
│   ├── push_to_hub.py              # Push fine-tuned model to HF Hub
│   └── export_onnx.py              # Export to ONNX for deployment
├── notebooks/
│   ├── 01_lora_finetuning_walkthrough.ipynb
│   ├── 02_qlora_llama2_on_single_gpu.ipynb
│   ├── 03_evaluation_and_bias_analysis.ipynb
│   └── 04_inference_optimization.ipynb
├── tests/
│   ├── test_trainer.py
│   ├── test_data_pipeline.py
│   └── test_inference.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── model_eval.yml          # Automated eval on new checkpoints
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### Python API — QLoRA fine-tuning on LLaMA 2 (single GPU, 8GB VRAM)

```python
from src.training.trainer import PEFTTrainer
from src.training.qlora_config import QLoRAConfig

config = QLoRAConfig(
    base_model="meta-llama/Llama-2-7b-hf",
    dataset_path="data/insurance_policies.jsonl",
    task="text-generation",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    max_seq_length=2048,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    output_dir="outputs/llama2-insurance-finetuned",
    mlflow_experiment="llama2-qlora-insurance-v1",
)

trainer = PEFTTrainer(config)
trainer.train()
trainer.evaluate()
trainer.save_adapter("outputs/llama2-insurance-adapter")
```

### CLI

```bash
# Train
python scripts/train.py --config configs/qlora_llama2_7b_instruct.yaml

# Evaluate on test set
python scripts/evaluate.py \
  --model-path outputs/llama2-insurance-finetuned \
  --dataset data/test.jsonl \
  --metrics rouge,bleu,bertscore

# Merge LoRA adapters into base model (for deployment)
python scripts/merge_adapters.py \
  --base meta-llama/Llama-2-7b-hf \
  --adapter outputs/llama2-insurance-adapter \
  --output outputs/llama2-insurance-merged

# Deploy to SageMaker
python src/deployment/sagemaker_deploy.py \
  --model-path outputs/llama2-insurance-merged \
  --instance-type ml.g5.xlarge \
  --endpoint-name llama2-insurance-prod
```

---

## 📊 GPU Memory — LoRA vs QLoRA vs Full Fine-Tuning

| Technique | Model | GPU VRAM | Trainable Params |
|---|---|---|---|
| Full fine-tuning | LLaMA-7B | ~56 GB | 7B (100%) |
| LoRA (r=16) | LLaMA-7B | ~18 GB | ~4M (0.06%) |
| QLoRA 4-bit (r=16) | LLaMA-7B | **~8 GB** ✅ | ~4M (0.06%) |
| QLoRA 4-bit (r=16) | LLaMA-13B | **~12 GB** ✅ | ~6M (0.05%) |
| QLoRA 4-bit (r=64) | Mistral-7B | **~10 GB** ✅ | ~16M (0.23%) |

---

## 🔬 Experiment Tracking (MLflow)

All runs auto-logged: `train/loss`, `eval/loss`, `eval/rouge1`, `eval/rouge2`, `eval/rougeL`, `eval/bleu`, `eval/bertscore`, `train/learning_rate`, GPU utilization, peak memory.

```bash
mlflow ui --port 5000
```

---

## 🚀 Deployment Options

| Target | Latency | Throughput | Cost |
|---|---|---|---|
| vLLM local server | ~50ms | High | GPU instance |
| AWS SageMaker RT | ~120ms | Auto-scaled | Pay-per-use |
| GCP Vertex AI | ~130ms | Auto-scaled | Pay-per-use |
| ONNX + TensorRT | ~20ms | Very high | GPU instance |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
