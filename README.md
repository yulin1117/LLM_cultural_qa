# LLM Cultural QA

This project focuses on fine-tuning Large Language Models for cultural question answering tasks.

## Project Structure

```
LLM_cultural_qa/
├── data/              # Dataset files
├── results/           # Model predictions and outputs
├── script/
│   ├── saq.py                    # Short Answer Question training script
│   ├── train_and_predict.py      # Main training and prediction pipeline
│   ├── cross_task/               # Cross-task transfer scripts
│   └── RAG/                      # Retrieval-Augmented Generation
│       └── search_context.py     # Context search and retrieval
└── README.md
```

## Setup

### Requirements

Install the required packages:

```bash
pip install torch transformers datasets trl peft bitsandbytes
pip install sentence-transformers faiss-cpu rank-bm25
pip install pandas numpy matplotlib tqdm
```

### Configuration

1. Update model paths and data paths in the scripts according to your setup
2. For RAG functionality, set your Serper API key in `script/RAG/search_context.py`

## Usage

### Training

For Short Answer Questions (SAQ):
```bash
python script/saq.py
```

### RAG Context Search

To augment your dataset with retrieved context:
```bash
python script/RAG/search_context.py
```

## Model

This project uses Meta-Llama-3-8B-Instruct with QLoRA fine-tuning for efficient training.

## License

[Add your license here]

## Citation

[Add citation information if applicable]
