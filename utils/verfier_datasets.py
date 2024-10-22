import json
import os
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List, Dict, Any
import transformers
from torch.utils.data import DataLoader
from dataclasses import dataclass

IGNORE_INDEX = -100


def read_jsonl(path: str) -> List[Dict]:
    """
    Read data from a JSONL file or JSON file.

    Args:
        path (str): Path to the input file

    Returns:
        List[Dict]: List of JSON objects from the file
    """
    try:
        with open(path) as fh:
            return [json.loads(line) for line in fh.readlines() if line]
    except:
        return json.load(open(path, "r", encoding="utf-8"))


def get_dataset_examples(data_paths: str) -> List[Dict]:
    """
    Load and combine examples from multiple data files.

    Args:
        data_paths (str): Comma-separated list of data file paths

    Returns:
        List[Dict]: Combined list of examples from all data files
    """
    examples = []
    for path in data_paths.split(","):
        examples.extend(read_jsonl(path))
    print(
        f"{len(examples)} examples, each with {len(examples[0]['outputs'])} solutions"
    )
    return examples


def create_train_dataloader(
    data_module: Dict[str, torch.utils.data.Dataset],
    training_args: dataclass,
) -> DataLoader:
    """
    Create a DataLoader for training data.

    Args:
        data_module (Dict): Dictionary containing dataset information
        training_args (dataclass): Training arguments including batch size

    Returns:
        DataLoader: DataLoader for training data
    """
    return DataLoader(
        data_module["train_dataset"],
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=data_module["train_dataset"].collate_fn,
    )


def create_test_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> DataLoader:
    """
    Create a DataLoader for testing data.

    Args:
        dataset (Dataset): The test dataset
        batch_size (int): Batch size for testing

    Returns:
        DataLoader: DataLoader for test data
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )


def create_verifier_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass
) -> Dict:
    """
    Create a data module for the verifier model.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text
        data_args (dataclass): Arguments for data processing

    Returns:
        Dict: Dictionary containing train and validation datasets
    """
    dataset_class = (
        ProcessVerifierDataset if data_args.process else OutcomeVerifierDataset
    )

    train_dataset = dataset_class(
        tokenizer=tokenizer,
        data_dir=data_args.data_dir,
        target_set=data_args.target_set,
        verifier_id=data_args.verifier_id,
        data_id=data_args.data_id,
        generator_id=data_args.generator_id,
        per_problem_sampling_solution=data_args.per_problem_sampling_solution,
    )

    return {"train_dataset": train_dataset, "val_dataset": None}


class OutcomeVerifierDataset(torch.utils.data.Dataset):
    """
    Dataset for outcome verification tasks with right padding.

    This dataset processes input-output pairs and their corresponding verification labels,
    handling tokenization and padding for model input.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_dir: str,
        target_set: str = None,
        per_problem_sampling_solution: int = None,
        loss_level: str = "token",
        loss_on_llm: bool = False,
    ):
        self.examples = get_dataset_examples(data_dir)
        assert len(self.examples[0]["outputs"]) >= per_problem_sampling_solution

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ("token", "step")

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self._process_examples(per_problem_sampling_solution)
        self._prepare_dataset()

    def _process_examples(self, per_problem_sampling_solution: int):
        """Process and deduplicate examples."""
        if per_problem_sampling_solution != -1:
            for example in self.examples:
                example["outputs"] = example["outputs"][:per_problem_sampling_solution]

        for ex in self.examples:
            responses = set()
            ex["outputs"] = [
                output
                for output in ex["outputs"]
                if output["response"] not in responses
                and not responses.add(output["response"])
            ]

    def _prepare_dataset(self):
        """Prepare dataset by tokenizing and filtering examples."""
        # Prepare indices and strings
        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for ex in self.examples]
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [
            [out["response"] for out in ex["outputs"]] for ex in self.examples
        ]
        v_classes = [
            [out["label"] == True for out in ex["outputs"]] for ex in self.examples
        ]

        # Flatten lists
        indices1 = [item for sublist in indices1 for item in sublist]
        indices2 = [item for sublist in indices2 for item in sublist]
        qns_str = [item for sublist in qns_str for item in sublist]
        solutions_str = [item for sublist in solutions_str for item in sublist]
        v_classes = [item for sublist in v_classes for item in sublist]

        # Tokenize
        qns_tokens = self.tokenizer(qns_str, padding=False).input_ids
        solutions_tokens = self.tokenizer(
            solutions_str, padding=False, add_special_tokens=False
        ).input_ids

        # Filter long sequences
        valid_indices = [
            i
            for i in range(len(qns_tokens))
            if len(qns_tokens[i]) + len(solutions_tokens[i]) + 1 <= 2048
        ]

        # Store processed data
        self.qns_tokens = [qns_tokens[i] for i in valid_indices]
        self.solutions_tokens = [solutions_tokens[i] for i in valid_indices]
        self.indices1 = [indices1[i] for i in valid_indices]
        self.indices2 = [indices2[i] for i in valid_indices]
        self.qns_str = [qns_str[i] for i in valid_indices]
        self.solutions_str = [solutions_str[i] for i in valid_indices]
        self.v_classes = [v_classes[i] for i in valid_indices]

        self.max_len = max(
            len(q) + len(s) + 1 for q, s in zip(self.qns_tokens, self.solutions_tokens)
        )
        print(f"Max tokens: {self.max_len}")
        print(f"Number of examples = {len(self.qns_str)}")
        self.n_question = len(self.examples)

    def __len__(self):
        return len(self.solutions_tokens)

    def __getitem__(self, idx):
        """Get a single item from the dataset with appropriate masking and labels."""
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        v_class = self.v_classes[idx]

        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]
        masks = ([0] * len(qn_tokens)) + ([1] * len(sol_tokens)) + [1]

        labels = mask_labels(input_ids, masks) if self.loss_on_llm else None
        v_labels = mask_labels([int(v_class)] * len(input_ids), masks)

        return {
            "idx1": self.indices1[idx],
            "idx2": self.indices2[idx],
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels) if labels else None,
            "v_labels": torch.tensor(v_labels),
            "qn_str": self.qns_str[idx],
            "qn_tokens": self.qns_tokens[idx],
            "sol_str": self.solutions_str[idx],
            "sol_tokens": self.solutions_tokens[idx],
            "v_class": self.v_classes[idx],
        }

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching examples.

        Args:
            instances (Sequence[Dict]): Batch of dataset items

        Returns:
            Dict[str, torch.Tensor]: Batched and padded tensors
        """
        batch = {
            key: [instance[key] for instance in instances]
            for key in instances[0].keys()
        }

        input_ids, attention_mask = right_pad_sequences(
            batch["input_ids"],
            padding_value=self.pad_token_id,
            return_attention_mask=True,
        )

        if self.loss_on_llm:
            batch["labels"] = right_pad_sequences(
                batch["labels"], padding_value=IGNORE_INDEX, return_attention_mask=False
            )

        batch["v_labels"] = right_pad_sequences(
            batch["v_labels"], padding_value=IGNORE_INDEX, return_attention_mask=False
        )

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask

        return batch


class ProcessVerifierDataset(OutcomeVerifierDataset):
    """
    Dataset for process verification tasks, extending OutcomeVerifierDataset.
    Handles process-based verification scores instead of binary outcomes.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        data_dir: str = "data/gsm8k/model_generation",
        target_set: str = None,
        generator_id: str = None,
        per_problem_sampling_solution: str = None,
    ):
        self.examples = get_dataset_examples(data_dir)
        assert len(self.examples[0]["outputs"]) >= per_problem_sampling_solution

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if per_problem_sampling_solution != -1:
            for example in self.examples:
                if "input" not in example:
                    example["input"] = example["question"]
                example["outputs"] = example["outputs"][:per_problem_sampling_solution]
        else:
            per_problem_sampling_solution = len(self.examples[0]["outputs"])

        for ex in self.examples:
            dedup_outputs = []
            responses = set()
            for output in ex["outputs"]:
                if output["response"] in responses:
                    continue
                responses.add(output["response"])
                dedup_outputs.append(output)
            ex["outputs"] = dedup_outputs

        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [
            [j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)
        ]
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [
            [outputs["response"] for outputs in ex["outputs"]] for ex in self.examples
        ]
        v_classes = [
            [outputs["process_vscores"] for outputs in ex["outputs"]]
            for ex in self.examples
        ]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        v_classes = self._flatten(v_classes)

        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        solutions_tokens = tokenizer(
            solutions_str, padding=False, add_special_tokens=False
        ).input_ids

        # Remove instances whose length is bigger than 2048
        (
            self.qns_tokens,
            self.solutions_tokens,
            self.indices1,
            self.indices2,
            self.qns_str,
            self.solutions_str,
            self.v_classes,
        ) = zip(
            *[
                (
                    qns_tokens[i],
                    solutions_tokens[i],
                    indices1[i],
                    indices2[i],
                    qns_str[i],
                    solutions_str[i],
                    v_classes[i],
                )
                for i in range(len(qns_tokens))
                if len(qns_tokens[i]) + len(solutions_tokens[i]) + 1 <= 2048
            ]
        )
        self.max_len = max(
            [
                len(qns_tokens[i]) + len(solutions_tokens[i]) + 1
                for i in range(len(solutions_tokens))
            ]
        )

        print(f"Max tokens: {self.max_len}")
        self.per_problem_sampling_solution = per_problem_sampling_solution
        print(f"Number of examples = {len(self.qns_str)}")
        self.n_question = len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        v_class = self.v_classes[idx]

        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]
        masks = ([0] * len(qn_tokens)) + ([1] * len(sol_tokens)) + ([1])

        labels = input_ids
        labels = mask_labels(labels, masks)

        v_class = [1] * len(qn_tokens) + v_class
        v_labels = mask_labels(v_class, masks)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        v_labels = torch.tensor(v_labels)
        return dict(
            idx1=self.indices1[idx],
            idx2=self.indices2[idx],
            input_ids=input_ids,
            labels=labels,
            v_labels=v_labels,
            qn_str=self.qns_str[idx],
            qn_tokens=self.qns_tokens[idx],
            sol_str=self.solutions_str[idx],
            sol_tokens=self.solutions_tokens[idx],
            v_class=self.v_classes[idx],
        )


def right_pad_sequences(
    sequences: List[torch.LongTensor],
    padding_value: int,
    return_attention_mask: bool = False,
) -> torch.Tensor:
    """
    Right pad sequences to the maximum length in the batch.

    Args:
        sequences: List of sequences to pad
        padding_value: Value to use for padding
        return_attention_mask: Whether to return attention mask

    Returns:
        Padded sequences and optionally attention mask
    """
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
    )
    if return_attention_mask:
        attention_mask = padded_sequences.ne(padding_value)
        return padded_sequences, attention_mask
    return padded_sequences


def mask_labels(labels: List[int], masks: List[bool]) -> List[int]:
    """
    Mask labels based on boolean mask.

    Args:
        labels: List of label values
        masks: List of boolean masks

    Returns:
        List of masked labels with IGNORE_INDEX for masked positions
    """
    assert len(labels) == len(masks)
    return [label if mask else IGNORE_INDEX for label, mask in zip(labels, masks)]
