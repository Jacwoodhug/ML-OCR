"""Training loop with mixed precision, CTC loss, LR scheduling, and logging."""

import os

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.crnn import CRNN
from src.model.ctc_decoder import greedy_decode
from src.training.metrics import batch_metrics


class Trainer:
    """Handles the full training loop for the CRNN OCR model."""

    def __init__(
        self,
        model: CRNN,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict | None = None,
    ):
        cfg = config or {}
        train_cfg = cfg.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        sched_cfg = train_cfg.get("scheduler", {})

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # CTC loss (blank = 0)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

        # Training params
        self.max_iterations = train_cfg.get("max_iterations", 500_000)
        self.val_interval = train_cfg.get("val_interval", 2000)
        self.checkpoint_interval = train_cfg.get("checkpoint_interval", 10000)
        self.log_interval = train_cfg.get("log_interval", 100)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 5.0)
        self.use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"

        # LR scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=sched_cfg.get("max_lr", 1e-3),
            total_steps=self.max_iterations,
            pct_start=sched_cfg.get("pct_start", 0.05),
        )

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Checkpointing
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        self.best_model_path = train_cfg.get("best_model_path", "checkpoints/best.pt")
        tag = train_cfg.get("checkpoint_tag", "")
        self.checkpoint_tag = f"_{tag}" if tag else ""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard
        tb_dir = train_cfg.get("tensorboard_dir", "runs")
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Tracking
        self.best_cer = float("inf")
        self.global_step = 0

    def train(self):
        """Run the full training loop."""
        self.model.train()
        train_iter = iter(self.train_loader)

        pbar = tqdm(total=self.max_iterations, desc="Training", dynamic_ncols=True)
        pbar.update(self.global_step)

        running_loss = 0.0
        log_steps = 0

        try:
            while self.global_step < self.max_iterations:
                # Get next batch (re-create iterator if exhausted)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss = self._train_step(batch)
                running_loss += loss
                log_steps += 1

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = running_loss / log_steps
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    running_loss = 0.0
                    log_steps = 0

                # Validation
                if self.val_loader and self.global_step % self.val_interval == 0:
                    val_metrics = self.validate()
                    self.writer.add_scalar("val/cer", val_metrics["cer"], self.global_step)
                    self.writer.add_scalar("val/wer", val_metrics["wer"], self.global_step)
                    self.writer.add_scalar("val/seq_acc", val_metrics["seq_acc"], self.global_step)
                    self.writer.add_scalar("val/loss", val_metrics["loss"], self.global_step)

                    tqdm.write(
                        f"Step {self.global_step}: "
                        f"val_loss={val_metrics['loss']:.4f} "
                        f"CER={val_metrics['cer']:.4f} "
                        f"WER={val_metrics['wer']:.4f} "
                        f"Acc={val_metrics['seq_acc']:.4f}"
                    )

                    # Save best model
                    if val_metrics["cer"] < self.best_cer:
                        self.best_cer = val_metrics["cer"]
                        self._save_checkpoint(self.best_model_path, is_best=True)
                        tqdm.write(f"  -> New best CER: {self.best_cer:.4f}")

                    self.model.train()

                # Periodic checkpoint
                if self.global_step % self.checkpoint_interval == 0:
                    path = os.path.join(self.checkpoint_dir, f"step_{self.global_step}{self.checkpoint_tag}.pt")
                    self._save_checkpoint(path)

        except KeyboardInterrupt:
            print("\nInterrupted — saving checkpoint...")
            path = os.path.join(self.checkpoint_dir, f"step_{self.global_step}{self.checkpoint_tag}.pt")
            self._save_checkpoint(path)
            print(f"Saved to {path}. Resume with --resume {path}")
        finally:
            pbar.close()
            self.writer.close()
            # Shut down persistent DataLoader workers so the process can exit
            if hasattr(self.train_loader, '_iterator') and self.train_loader._iterator is not None:
                self.train_loader._iterator._shutdown_workers()
            if self.val_loader and hasattr(self.val_loader, '_iterator') and self.val_loader._iterator is not None:
                self.val_loader._iterator._shutdown_workers()

        if self.global_step >= self.max_iterations:
            print(f"Training complete. Best CER: {self.best_cer:.4f}")

    def _train_step(self, batch: dict) -> float:
        """Execute a single training step."""
        images = batch["images"].to(self.device)
        targets = batch["targets"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)
        image_widths = batch["image_widths"].to(self.device)

        self.optimizer.zero_grad()

        with autocast("cuda", enabled=self.use_amp):
            log_probs = self.model(images)  # (T, B, C)
            input_lengths = self.model.compute_input_lengths(image_widths)

            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Only advance scheduler if the optimizer actually ran (GradScaler
        # skips the step when gradients overflow, e.g. on the first iteration)
        if self.scaler.get_scale() >= scale_before:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and return metrics."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            image_widths = batch["image_widths"].to(self.device)
            texts = batch["texts"]

            with autocast("cuda", enabled=self.use_amp):
                log_probs = self.model(images)
                input_lengths = self.model.compute_input_lengths(image_widths)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions
            preds = greedy_decode(log_probs.float())
            all_preds.extend(preds)
            all_targets.extend(texts)

        avg_loss = total_loss / max(num_batches, 1)
        metrics = batch_metrics(all_preds, all_targets)
        metrics["loss"] = avg_loss

        # Log a few sample predictions
        n_samples = min(5, len(all_preds))
        for i in range(n_samples):
            self.writer.add_text(
                f"val/sample_{i}",
                f"Target: `{all_targets[i]}`  \nPred: `{all_preds[i]}`",
                self.global_step,
            )

        return metrics

    def _save_checkpoint(self, path: str, is_best: bool = False):
        """Save model and training state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_cer": self.best_cer,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["step"]
        self.best_cer = checkpoint.get("best_cer", float("inf"))
        print(f"Resumed from step {self.global_step} (best CER: {self.best_cer:.4f})")
