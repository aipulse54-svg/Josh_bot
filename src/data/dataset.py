"""Dataset loading and preprocessing utilities."""
import logging
from typing import Dict, List, Optional
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CodeMathDataset:
    """Handles loading and preprocessing code and math datasets."""

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        num_workers: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers

    def load_datasets(self) -> tuple:
        """Load code and math datasets from HuggingFace."""
        logger.info("Loading datasets...")
        
        try:
            # Load code dataset
            code_dataset = load_dataset(
                "codeparrot/codeparrot",
                split="train",
                streaming=False,
                trust_remote_code=True,
            )
            logger.info(f"Loaded code dataset with {len(code_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not load codeparrot: {e}. Using alternative...")
            code_dataset = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                split="train",
            )

        try:
            # Load math dataset
            math_dataset = load_dataset(
                "hendrycks/competition_math",
                split="train",
                trust_remote_code=True,
            )
            logger.info(f"Loaded math dataset with {len(math_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not load competition_math: {e}")
            math_dataset = None

        return code_dataset, math_dataset

    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize text examples."""
        # Handle different dataset formats
        if "code" in examples:
            text = examples["code"]
        elif "problem" in examples:
            text = examples["problem"] + "\n" + examples.get("solution", "")
        elif "text" in examples:
            text = examples["text"]
        else:
            # Fallback for unknown formats
            text = str(list(examples.values())[0])

        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # Create labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def preprocess_dataset(
        self,
        dataset: Dataset,
        dataset_name: str = "dataset",
    ) -> Dataset:
        """Preprocess dataset with tokenization."""
        logger.info(f"Preprocessing {dataset_name}...")
        
        # Limit dataset size for faster processing in demo
        if len(dataset) > 50000:
            dataset = dataset.select(np.random.choice(len(dataset), 50000, replace=False))
        
        processed = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {dataset_name}",
        )

        return processed

    def prepare_datasets(
        self,
        train_split: float = 0.9,
    ) -> tuple:
        """Prepare train and validation datasets."""
        code_dataset, math_dataset = self.load_datasets()

        # Preprocess datasets
        processed_code = self.preprocess_dataset(code_dataset, "code_dataset")
        
        if math_dataset:
            processed_math = self.preprocess_dataset(math_dataset, "math_dataset")
            # Combine datasets
            combined_dataset = concatenate_datasets([processed_code, processed_math])
        else:
            combined_dataset = processed_code

        # Split into train and validation
        split_dataset = combined_dataset.train_test_split(
            test_size=1 - train_split,
            seed=42,
        )

        logger.info(
            f"Final train size: {len(split_dataset['train'])}, "
            f"val size: {len(split_dataset['test'])}"
        )

        return split_dataset["train"], split_dataset["test"]
