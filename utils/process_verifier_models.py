from typing import Optional, List, Dict, Any, Union, Mapping
import transformers
from transformers.generation.utils import ModelOutput
from torch import nn
import torch.nn.functional as F
import torch
import os
from dataclasses import dataclass
from accelerate import Accelerator


@dataclass
class VerifierModelOutput(ModelOutput):
    """
    Output type for the Verifier model.

    Args:
        loss (Optional[torch.FloatTensor]): Combined loss from verifier and language model
        v_scores (torch.FloatTensor): Verification scores for input sequences
        all_losses (Optional[Dict[str, torch.FloatTensor]]): Detailed breakdown of losses
        seq_v_scores (Optional[Dict[str, torch.FloatTensor]]): Sequence-level verification scores
    """

    loss: Optional[torch.FloatTensor] = None
    v_scores: torch.FloatTensor = None
    all_losses: Optional[Dict[str, torch.FloatTensor]] = None
    seq_v_scores: Optional[Dict[str, torch.FloatTensor]] = None


class Verifier(nn.Module):
    """
    Neural verifier model that builds upon a pre-trained language model backbone.

    Adds verification capabilities to assess the quality or correctness of generated text.

    Args:
        backbone: Pre-trained language model to use as the backbone
        checkpoint_dir (Optional[str]): Directory containing checkpoint files
        torch_dtype: Torch data type for model parameters (default: torch.bfloat16)
    """

    def __init__(self, backbone, checkpoint_dir=None, torch_dtype=torch.bfloat16):
        super(Verifier, self).__init__()
        self.backbone = backbone

        # Transformation parameters
        self.gain = nn.Parameter(
            torch.randn(
                1,
            )
        )
        self.bias = nn.Parameter(
            torch.randn(
                1,
            )
        )
        self.dropout = nn.Dropout(p=0.2)
        self.vscore_head = nn.Linear(
            self.backbone.get_input_embeddings().embedding_dim, 1, bias=False
        )

        if checkpoint_dir and os.path.exists(
            os.path.join(checkpoint_dir, "verifier.pth")
        ):
            verifier_params = torch.load(os.path.join(checkpoint_dir, "verifier.pth"))
            self.load_state_dict(verifier_params, strict=False)
        else:
            self.init_head_params()

        self.to(dtype=torch_dtype)
        self.pad_token_id = backbone.config.pad_token_id

    def init_head_params(self):
        """Initialize the verification head using average of output embeddings."""
        output_embeddings = self.backbone.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings.mean(dim=0, keepdim=True)
        self.vscore_head.weight = nn.Parameter(output_embeddings_avg)

    def loss_fct(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        """Calculate MSE loss with mask for verification scores."""
        return mse_loss_with_mask(v_scores.squeeze(), v_labels.type_as(v_scores))

    def transform(self, last_hidden_states):
        """Apply linear transformation to hidden states."""
        return self.gain * last_hidden_states + self.bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        v_labels: Optional[torch.LongTensor] = None,
        output_all_losses: Optional[bool] = None,
    ) -> VerifierModelOutput:
        """
        Forward pass of the verifier model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input sequence
            position_ids: Position IDs for input tokens
            past_key_values: Cached key values for efficient inference
            labels: Labels for language modeling loss
            v_labels: Labels for verification loss
            output_all_losses: Whether to output detailed loss breakdown

        Returns:
            VerifierModelOutput containing loss and verification scores
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        llm_logits = outputs.logits
        llm_loss = outputs.loss
        llm_hidden_states = outputs.hidden_states

        v_hidden_states = self.transform(llm_hidden_states[-1])
        v_scores = self.vscore_head(self.dropout(v_hidden_states))

        v_loss, loss = None, None
        if v_labels is not None:
            v_loss = self.loss_fct(v_scores, v_labels)
            loss = v_loss + (llm_loss if labels is not None else 0)

        all_losses = None
        if output_all_losses:
            all_losses = {"llm_loss": llm_loss, "v_loss": v_loss}

        return VerifierModelOutput(
            loss=loss,
            v_scores=v_scores,
            all_losses=all_losses,
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing in the backbone model."""
        self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing in the backbone model."""
        self.backbone.gradient_checkpointing_disable()


def mse_loss_with_mask(scores: torch.FloatTensor, labels: torch.FloatTensor):
    """
    Calculate MSE loss while ignoring specified padding indices.

    Args:
        scores: Model prediction scores
        labels: Ground truth labels

    Returns:
        Masked MSE loss averaged over batch
    """
    IGNORE_INDEX = -100  # Moved constant here for clarity
    scores = torch.where(labels.ne(IGNORE_INDEX), scores, 0)
    labels = torch.where(labels.ne(IGNORE_INDEX), labels, 0)
    return F.mse_loss(scores, labels, reduction="sum") / scores.shape[0]


def save_verifier(
    accelerator: Accelerator,
    model: transformers.AutoModelForCausalLM,
    cpu_state_dict: Mapping,
    output_dir: str,
):
    """
    Save verifier model state to disk.

    Args:
        accelerator: Hugging Face Accelerator instance
        model: The verifier model
        cpu_state_dict: Model state dict on CPU
        output_dir: Directory to save the model
    """
    cpu_state_dict_backbone = {
        k.split("backbone.")[1]: v
        for k, v in cpu_state_dict.items()
        if k.startswith("backbone")
    }
    cpu_state_dict_verifier = {
        k: v for k, v in cpu_state_dict.items() if not k.startswith("backbone")
    }
    accelerator.unwrap_model(model).backbone.save_pretrained(
        output_dir,
        state_dict=cpu_state_dict_backbone,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    accelerator.save(cpu_state_dict_verifier, os.path.join(output_dir, "verifier.pth"))


def build_verifier_from_scratch(
    model_args: dataclass, training_args: dataclass, accelerator: Accelerator
):
    """
    Build a new verifier model from scratch.

    Args:
        model_args: Model configuration arguments
        training_args: Training configuration arguments
        accelerator: Hugging Face Accelerator instance

    Returns:
        Tuple of (verifier model, tokenizer)
    """
    backbone, tokenizer = build_model(model_args, training_args, accelerator)
    return Verifier(backbone).to(accelerator.device), tokenizer


def build_verifier_from_osv(
    model_args: dataclass, training_args: dataclass, accelerator: Accelerator
):
    """
    Build a verifier model from a pre-trained checkpoint.

    Args:
        model_args: Model configuration arguments
        training_args: Training configuration arguments
        accelerator: Hugging Face Accelerator instance

    Returns:
        Tuple of (verifier model, tokenizer)
    """
    v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=training_args.cache_dir,
    )

    if training_args.gradient_checkpointing:
        v_backbone.gradient_checkpointing_enable()

    verifier = Verifier(v_backbone, checkpoint_dir=model_args.model_name_or_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=False,
    )

    return verifier.to(accelerator.device), tokenizer


def load_generator_and_verifier(model_args: dataclass):
    """
    Load both generator and verifier models from checkpoints.

    Args:
        model_args: Model configuration arguments

    Returns:
        Tuple of (generator model, verifier model, tokenizer)
    """
    generator, tokenizer = load_model(model_args)

    v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.verifier_model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    verifier = Verifier(
        v_backbone, checkpoint_dir=model_args.verifier_model_name_or_path
    )
    return generator, verifier, tokenizer
