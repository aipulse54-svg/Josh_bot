"""LLM Model classes for training and inference."""
import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class LLMAssistant:
    """Main LLM Assistant class for code generation and math solving."""

    def __init__(
        self,
        model_name: str = "gpt2",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = lora_config or {}

        logger.info(f"Loading model: {model_name} on device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA if enabled
        if use_lora:
            self._apply_lora()

        self.model.to(device)
        logger.info(f"Model loaded successfully. Trainable params: {self._count_parameters()}")

    def _apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        logger.info("Applying LoRA configuration...")
        
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 8),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            target_modules=self.lora_config.get("target_modules", ["c_attn"]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> list:
        """Generate text completions.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        self.model.eval()
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts

    def get_model(self) -> PreTrainedModel:
        """Return the underlying model."""
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the tokenizer."""
        return self.tokenizer
