"""Training script for LLM Assistant."""
import os
import logging
import yaml
import argparse
from pathlib import Path
import torch

from src.model.llm import LLMAssistant
from src.data.dataset import CodeMathDataset
from src.training.trainer import LLMTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    """Main training function."""
    logger.info("Starting LLM Assistant training...")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")
    
    # Create output directory
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    
    # Initialize tokenizer and dataset handler
    from transformers import AutoTokenizer
    
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset_handler = CodeMathDataset(
        tokenizer=tokenizer,
        max_seq_length=config["dataset"]["max_seq_length"],
        num_workers=config["dataset"]["preprocessing_num_workers"],
    )
    
    logger.info("Loading datasets...")
    train_dataset, val_dataset = dataset_handler.prepare_datasets()
    
    # Initialize model
    logger.info("Initializing model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model = LLMAssistant(
        model_name=model_name,
        use_lora=config["lora"]["enabled"],
        lora_config=config["lora"],
        device=device,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LLMTrainer(
        model=llm_model.get_model(),
        tokenizer=llm_model.get_tokenizer(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config,
    )
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.train()
    
    logger.info("Training completed!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    main(args)
