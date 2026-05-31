# LLM Assistant - Production-Ready AI System

A comprehensive, production-ready Large Language Model (LLM) assistant designed for code generation and mathematical problem solving. Built with PyTorch, Hugging Face Transformers, and advanced optimization techniques.

## 🚀 Features

### Core Capabilities
- **Code Generation**: Generate code in multiple programming languages
- **Math Solving**: Symbolic algebra and equation solving using SymPy
- **Code Explanation**: Understand and explain code snippets
- **General Q&A**: Answer questions on various topics

### Advanced Techniques
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with minimal parameters
- **Mixed Precision Training**: FP16 support for faster training and lower memory usage
- **Gradient Accumulation**: Support for larger effective batch sizes
- **Evaluation during Training**: Monitor model performance continuously
- **Model Checkpointing**: Save best models and resume training
- **Weights & Biases Integration**: Comprehensive experiment tracking

### Production Ready
- **FastAPI Server**: REST API for serving the model
- **Pydantic Validation**: Type-safe request/response handling
- **Comprehensive Logging**: Debug and monitor in production
- **Error Handling**: Robust error recovery
- **Configuration Management**: YAML-based configuration

## 📋 Project Structure

```
Josh_bot/
├── config/
│   └── config.yaml                 # Training and model configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── dataset.py             # Dataset loading and preprocessing
│   ├── model/
│   │   └── llm.py                 # LLM model class
│   ├── training/
│   │   └── trainer.py             # Training loop
│   ├── inference/
│   │   └── assistant.py           # Inference and task-specific logic
│   └── api/
│       └── server.py              # FastAPI server
├── notebooks/
│   └── train_colab.ipynb          # Google Colab training notebook
├── scripts/
│   ├── train.py                   # Training script
│   └── inference.py               # Inference script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/aipulse54-svg/Josh_bot.git
cd Josh_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Run cells sequentially
3. The notebook handles all setup automatically

## 📚 Training

### Configuration

Edit `config/config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Dataset sources
- LoRA settings
- Quantization options

### Local Training

```bash
python scripts/train.py --config config/config.yaml
```

### Google Colab Training

1. Upload `notebooks/train_colab.ipynb` to Colab
2. Set up Weights & Biases for monitoring
3. Run cells to train on free GPUs
4. Model is automatically saved to Google Drive

## 🎯 Inference

### Using the Assistant Class

```python
from src.model.llm import LLMAssistant
from src.inference.assistant import CodeMathAssistant

# Load model
llm = LLMAssistant(model_name="gpt2")
assistant = CodeMathAssistant(llm)

# Generate code
result = assistant.generate_code(
    "Write a function to sort a list",
    language="python"
)
print(result["code"])

# Solve math problem
result = assistant.solve_math_problem("Solve: 2x + 5 = 13")
print(result["solutions"])
```

### Using the REST API

```bash
# Start server
python -m uvicorn src.api.server:app --reload
```

```bash
# Generate code
curl -X POST "http://localhost:8000/generate-code" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to calculate factorial",
    "language": "python"
  }'

# Solve math problem
curl -X POST "http://localhost:8000/solve-math" \
  -H "Content-Type: application/json" \
  -d '{"problem": "Solve: x^2 = 16"}'
```

### Command Line Inference

```bash
# Generate code
python scripts/inference.py \
  --task code \
  --prompt "Write a function to reverse a string" \
  --language python

# Solve math problem
python scripts/inference.py \
  --task math \
  --prompt "Solve: 3x - 2 = 10"

# Explain code
python scripts/inference.py \
  --task explain \
  --prompt "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
```

## 📊 Monitoring Training

### Weights & Biases

1. Sign up at [wandb.ai](https://wandb.ai)
2. Login in your training environment: `wandb login`
3. Access your dashboard for real-time metrics

### TensorBoard

```bash
tensorboard --logdir ./outputs/checkpoint/runs
```

## 🛠️ Advanced Configuration

### Using Different Models

In `config/config.yaml`:
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # or any HuggingFace model
```

### Quantization for Memory Efficiency

Enable in `config/config.yaml`:
```yaml
quantization:
  enabled: true
  load_in_4bit: true
```

### Distributed Training

For multi-GPU training, use `accelerate`:
```bash
acccelerate config  # Configure your setup
acccelerate launch scripts/train.py --config config/config.yaml
```

## 📈 Performance Tips

1. **Reduce batch size** if running out of memory
2. **Enable gradient accumulation** for effective larger batches
3. **Use LoRA** for efficient fine-tuning
4. **Enable mixed precision (FP16)** for faster training
5. **Use smaller models** for faster iteration

## 🐛 Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable quantization
- Use a smaller model

### Slow Training
- Enable mixed precision (FP16)
- Reduce dataset size for testing
- Use faster tokenization

### Import Errors
```bash
pip install --upgrade transformers datasets torch
```

## 📝 Example Use Cases

### Code Generation
```python
result = assistant.generate_code(
    "Create a class for managing a todo list with add, remove, and list methods",
    language="python"
)
```

### Math Solving
```python
result = assistant.solve_math_problem(
    "Solve the equation: x^2 - 5x + 6 = 0"
)
```

### Code Explanation
```python
code = """
class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop() if not self.is_empty() else None
"""
result = assistant.explain_code(code)
```

## 📜 License

MIT License - feel free to use in your projects

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Support

For issues and questions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Models from [Hugging Face](https://huggingface.co/)
- Training tracked with [Weights & Biases](https://wandb.ai/)
- Math solving with [SymPy](https://www.sympy.org/)

---

**Happy training and inferencing! 🚀**
