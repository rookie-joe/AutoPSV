from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import gc
import wandb
import transformers
from accelerator import Accelerator
from tqdm.auto import tqdm

from utils.process_verifier_models import (
    save_verifier,
    build_verifier_from_osv,
    build_verifier_from_scratch,
)
from utils.states import set_training_states, set_random_seed, get_optimizers
from utils.verifier_datasets import (
    make_training_verifier_data_module,
    make_training_dataloaders,
)


@dataclass
class ModelParams:
    """Parameters for model configuration."""

    model_name_or_path: Optional[str] = field(
        default="none",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co"
        },
    )


@dataclass
class DataParams:
    """Parameters for data handling and processing."""

    data_dir: str = field(
        default="none", metadata={"help": "Path to the training data directory"}
    )
    data_id: str = field(
        default="none", metadata={"help": "Identifier for the dataset"}
    )
    target_set: str = field(
        default="train", metadata={"help": "Target dataset split for training"}
    )
    val_target_set: Optional[str] = field(
        default=None, metadata={"help": "Target dataset split for validation"}
    )
    generator_id: str = field(
        default="none", metadata={"help": "Identifier for the generator model"}
    )
    per_problem_sampling_solution: int = field(
        default=-1,
        metadata={"help": "Number of solutions to sample per problem. -1 for all"},
    )
    loss_level: str = field(
        default="token",
        metadata={"help": "Level at which to compute loss: 'token' or 'sequence'"},
    )
    loss_on_llm: bool = field(
        default=False,
        metadata={"help": "Whether to compute loss on language model outputs"},
    )
    dedup: bool = field(default=False, metadata={"help": "Whether to deduplicate data"})
    process: bool = field(
        default=False, metadata={"help": "Whether to preprocess data"}
    )
    verifier_id: str = field(
        default="none", metadata={"help": "Identifier for the verifier model"}
    )


@dataclass
class TrainParams:
    """Parameters for training configuration."""

    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization"}
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of training steps. Overrides num_train_epoches"
        },
    )
    num_train_epoches: int = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    eval_steps: int = field(
        default=-1,
        metadata={"help": "Evaluation frequency in steps. Overrides eval_epoches"},
    )
    eval_epoches: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    per_device_eval_batch_size: int = field(default=4)
    resume_from_checkpoint: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0)
    lr_scheduler_type: str = field(
        default="linear", metadata={"help": "Learning rate scheduler type"}
    )
    warmup_steps: int = field(
        default=-1, metadata={"help": "Number of warmup steps. Overrides warmup_ratio"}
    )
    warmup_ratio: float = field(default=0)
    num_lr_epoches_fs: int = field(default=-1)
    num_lr_epoches_scatter: int = field(default=-1)
    logging_steps: int = field(
        default=-1,
        metadata={"help": "Logging frequency in steps. Overrides logging_epoches"},
    )
    logging_epoches: int = field(default=1)
    save_steps: int = field(
        default=-1,
        metadata={"help": "Saving frequency in steps. Overrides save_epoches"},
    )
    save_epoches: int = field(default=1)
    save_total_limit: int = field(default=3)
    save_best: bool = field(default=False)
    fp16: bool = field(default=False)
    seed: int = field(default=42)
    resume: bool = field(default=False)

    @property
    def num_training_steps(self) -> int:
        """Calculate total number of training steps."""
        return self.max_steps if self.max_steps > 0 else self.num_train_epoches

    @property
    def num_logging_steps(self) -> int:
        """Calculate number of steps between logging."""
        return self.logging_steps if self.logging_steps > 0 else self.logging_epoches

    @property
    def per_save_steps(self) -> int:
        """Calculate number of steps between saves."""
        return self.save_steps if self.save_steps > 0 else self.save_epoches


@dataclass
class OutputParams:
    """Parameters for output and logging configuration."""

    logging_dir: str = field(
        default="wandb/", metadata={"help": "Directory for wandb logs"}
    )
    save_dir: str = field(
        default="checkpoints/",
        metadata={"help": "Directory for saving model checkpoints"},
    )


class TrainingManager:
    """Manages the training process for the verifier model."""

    def __init__(
        self,
        model_args: ModelParams,
        data_args: DataParams,
        training_args: TrainParams,
        output_args: OutputParams,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.output_args = output_args
        self.config_args_dict = self._prepare_config_dict()

        set_random_seed(training_args.seed)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_args.gradient_accumulation_steps
        )

    def _prepare_config_dict(self) -> Dict[str, Any]:
        """Prepare configuration dictionary from all arguments."""
        config = self.model_args.__dict__.copy()
        config.update(self.data_args.__dict__)
        config.update(self.training_args.__dict__)
        return config

    def setup_model_and_data(self):
        """Initialize model, tokenizer, and data loaders."""
        # Build or load model
        if self.model_args.model_name_or_path and os.path.exists(
            os.path.join(self.model_args.model_name_or_path, "verifier.pth")
        ):
            self.model, self.tokenizer = build_verifier_from_osv(
                self.model_args, self.training_args, self.accelerator
            )
        else:
            self.model, self.tokenizer = build_verifier_from_scratch(
                self.model_args, self.training_args, self.accelerator
            )

        # Prepare data
        data_module = make_training_verifier_data_module(self.tokenizer, self.data_args)
        self.train_dataloader = make_training_dataloaders(
            data_module, self.training_args
        )

        # Setup training
        set_training_states(data_module, self.training_args)
        self.optimizer, self.lr_scheduler = get_optimizers(
            self.model, self.training_args
        )

        # Prepare for acceleration
        self.model, self.train_dataloader, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.train_dataloader, self.optimizer, self.lr_scheduler
            )
        )

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.accelerator.is_main_process:
            project_name = os.environ["WANDB_PROJECT"]
            logging_dir = os.path.join(self.output_args.logging_dir, project_name)
            os.makedirs(logging_dir, exist_ok=True)
            wandb_id = self.output_args.save_dir
            wandb.init(id=wandb_id, dir=logging_dir, config=self.config_args_dict)

    def _load_checkpoint(self):
        """Load model from checkpoint if specified."""
        loaded_step = -1
        loaded_step_dir = ""

        if self.training_args.resume_from_checkpoint:
            assert os.path.exists(self.output_args.save_dir)
            subdirs = [
                d
                for d in os.listdir(self.output_args.save_dir)
                if os.path.isdir(os.path.join(self.output_args.save_dir, d))
            ]

            for subdir in subdirs:
                try:
                    step = int(subdir)
                    if step > loaded_step:
                        loaded_step = step
                        loaded_step_dir = subdir
                except ValueError:
                    continue

            assert loaded_step
            loaded_step_dir_path = os.path.join(
                self.output_args.save_dir, loaded_step_dir
            )
            print(f"Loading model from: {loaded_step_dir_path}")
            self.accelerator.load_state(loaded_step_dir_path)
            loaded_step *= self.training_args.gradient_accumulation_steps

        return loaded_step

    def _log_metrics(self, loss: float, all_losses: Dict[str, float], step: int):
        """Log metrics to Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.log(
                {
                    "loss": loss,
                    "v_loss": all_losses.get("v_loss").item(),
                    "llm_loss": (
                        all_losses.get("llm_loss").item()
                        if self.data_args.loss_on_llm
                        else 0
                    ),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                },
                step=step,
            )

    def _save_checkpoint(self, global_step: int):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        resume_dir = os.path.join(
            self.output_args.save_dir,
            str(global_step // self.training_args.gradient_accumulation_steps),
        )
        print(f"Saving model in {resume_dir}")
        self.accelerator.save_state(resume_dir)

    def train(self):
        """Execute the training loop."""
        self._init_wandb()
        start_global_step = self._load_checkpoint()

        global_step = 0
        cur_epoch = 0
        self.model.train()

        while global_step < self.training_args.num_training_steps:
            train_dataloader_iterator = (
                tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc="Training",
                )
                if self.accelerator.is_main_process
                else enumerate(self.train_dataloader)
            )

            for local_step, batch in train_dataloader_iterator:
                if global_step < start_global_step:
                    global_step += 1
                    continue

                batch_input = {
                    k: v
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask", "labels", "v_labels")
                }

                # Forward and backward pass
                with self.accelerator.autocast(), self.accelerator.accumulate(
                    self.model
                ):
                    output = self.model(**batch_input, output_all_losses=True)
                    loss = output.loss
                    all_losses = output.all_losses
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.accelerator.wait_for_everyone()

                # Update progress bar
                if self.accelerator.is_main_process:
                    train_dataloader_iterator.set_postfix(
                        epoch=cur_epoch,
                        step=local_step,
                        loss=loss.item(),
                        v_loss=all_losses.get("v_loss").item(),
                        llm_loss=(
                            all_losses.get("llm_loss").item()
                            if self.data_args.loss_on_llm
                            else 0
                        ),
                    )

                # Log metrics
                if (
                    global_step % self.training_args.gradient_accumulation_steps
                    and (global_step % self.training_args.gradient_accumulation_steps)
                    % self.training_args.num_logging_steps
                    == 0
                ):
                    self._log_metrics(loss.item(), all_losses, global_step)

                # Save checkpoint
                if (
                    global_step != 0
                    and (
                        global_step % self.training_args.gradient_accumulation_steps
                        == 0
                    )
                    and (global_step // self.training_args.gradient_accumulation_steps)
                    % self.training_args.per_save_steps
                    == 0
                    and global_step != start_global_step
                ):
                    self._save_checkpoint(global_step)

                global_step += 1

            cur_epoch += 1
            del train_dataloader_iterator
            gc.collect()
            self.accelerator.wait_for_everyone()

        # Save final model
        self.accelerator.wait_for_everyone()
        save_verifier(
            self.accelerator, self.model, self.tokenizer, self.output_args.save_dir
        )

        if self.accelerator.is_main_process:
            wandb.finish()


def main():
    """Main function to setup and run training."""
    parser = transformers.HfArgumentParser(
        (ModelParams, DataParams, TrainParams, OutputParams)
    )
    model_args, data_args, training_args, output_args = (
        parser.parse_args_into_dataclasses()
    )

    trainer = TrainingManager(model_args, data_args, training_args, output_args)
    trainer.setup_model_and_data()
    trainer.train()


if __name__ == "__main__":
    main()
