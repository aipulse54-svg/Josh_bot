"""Custom trainer for LLM model."""
import logging
from typing import Optional, Dict, Any
import torch
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb

logger = logging.getLogger(__name__)


class LLMTrainer:
    """Wrapper class for training LLM models."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments."""
        training_config = self.config.get("training", {})
        
        return TrainingArguments(
            output_dir=training_config.get("output_dir", "./outputs"),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 2e-4),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            warmup_steps=training_config.get("warmup_steps", 500),
            weight_decay=training_config.get("weight_decay", 0.01),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            logging_steps=training_config.get("logging_steps", 100),
            eval_steps=training_config.get("eval_steps", 500),
            save_steps=training_config.get("save_steps", 500),
            save_total_limit=training_config.get("save_total_limit", 3),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=training_config.get("fp16", torch.cuda.is_available()),
            report_to=["wandb"] if self.config.get("logging", {}).get("use_wandb") else [],
            seed=training_config.get("seed", 42),
        )

    def train(self) -> Dict[str, Any]:
        """Train the model."""
        logger.info("Starting training...")
        
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        
        return train_result.metrics
