"""
ResumableTrainer — A plug-and-play training utility for Google Colab
Handles checkpointing, state persistence, and seamless resume across sessions.

Usage:
    trainer = ResumableTrainer(
        project_name="Cifar_10",
        experiment_name="model_1",
        model_fn=create_model,         # function that returns a compiled model
        checkpoint_root="/content/drive/MyDrive/Colab_Experiments"
    )
    trainer.fit(train_data, val_data, epochs=100, batch_size=64)
"""

import json
import glob
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, Callback
)


# ─────────────────────────────────────────────
# State persistence callback
# ─────────────────────────────────────────────

class TrainingStateCallback(Callback):
    """
    Saves full training state after every epoch.
    Tracks: current epoch, best val metric, early stopping counter, completion flag.
    """

    def __init__(self, state_path, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.state_path = Path(state_path)
        self.monitor = monitor
        self.mode = mode
        self.state = {}

    def set_state(self, state: dict):
        """Load existing state (called before training begins)."""
        self.state = state.copy()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val = logs.get(self.monitor, None)

        # Update best val metric
        if current_val is not None:
            best = self.state.get('best_val_metric', None)
            if best is None:
                self.state['best_val_metric'] = float(current_val)
            else:
                if self.mode == 'max' and current_val > best:
                    self.state['best_val_metric'] = float(current_val)
                    self.state['patience_counter'] = 0
                elif self.mode == 'min' and current_val < best:
                    self.state['best_val_metric'] = float(current_val)
                    self.state['patience_counter'] = 0
                else:
                    self.state['patience_counter'] = self.state.get('patience_counter', 0) + 1

        self.state['last_epoch'] = epoch + 1  # 1-indexed (next epoch to run)
        self.state['last_updated'] = datetime.now().isoformat()
        self.state['training_complete'] = False

        self._atomic_save()

    def on_train_end(self, logs=None):
        self.state['training_complete'] = True
        self.state['last_updated'] = datetime.now().isoformat()
        self._atomic_save()
        print(f"\n Training state saved → {self.state_path}")

    def _atomic_save(self):
        """Write to a temp file first, then rename — prevents corruption on crash."""
        tmp = self.state_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.state_path)


# ─────────────────────────────────────────────
# Stateful EarlyStopping
# ─────────────────────────────────────────────

class StatefulEarlyStopping(EarlyStopping):
    """
    EarlyStopping that restores its internal counter and best value
    from a saved state — so patience carries over across sessions.
    """

    def __init__(self, saved_best=None, saved_patience_counter=0, **kwargs):
        super().__init__(**kwargs)
        self._saved_best = saved_best
        self._saved_patience_counter = saved_patience_counter

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # Restore internal state if available
        if self._saved_best is not None:
            self.best = self._saved_best
            self.wait = self._saved_patience_counter
            print(f" EarlyStopping restored — best={self.best:.4f}, patience_counter={self.wait}")


# ─────────────────────────────────────────────
# Core ResumableTrainer
# ─────────────────────────────────────────────

class ResumableTrainer:
    """
    A plug-and-play resumable training utility for Google Colab.

    Features:
    - Auto-detects and resumes from latest checkpoint
    - Persists full training state (epoch, best metric, patience counter)
    - Stateful EarlyStopping that carries over across sessions
    - CSV logging with append support
    - Works across multiple Colab accounts via shared Google Drive

    Args:
        project_name (str):     Top-level folder name (e.g., "Cifar_10")
        experiment_name (str):  Sub-folder / model name (e.g., "model_1")
        model_fn (callable):    Function that returns a freshly compiled Keras model
        checkpoint_root (str):  Root path on Google Drive
        monitor (str):          Metric to monitor (default: 'val_accuracy')
        mode (str):             'max' or 'min' depending on monitor metric
        patience (int):         EarlyStopping patience (default: 7)
        save_freq (str/int):    How often to save epoch checkpoints (default: 'epoch')
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        model_fn: callable,
        checkpoint_root: str = "/content/drive/MyDrive/Colab_Experiments",
        monitor: str = "val_accuracy",
        mode: str = "max",
        patience: int = 7,
        save_freq="epoch",
    ):
        self.experiment_name = experiment_name
        self.model_fn = model_fn
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.save_freq = save_freq

        # Directory setup
        self.ckpt_dir = Path(checkpoint_root) / project_name / experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.best_model_path = self.ckpt_dir / f"{experiment_name}_best.keras"
        self.epoch_template    = str(self.ckpt_dir / f"{experiment_name}_epoch_{{epoch:04d}}.keras")
        self.state_path        = self.ckpt_dir / "training_state.json"
        self.csv_log_path      = self.ckpt_dir / "training_log.csv"

        # Will be populated on fit()
        self.model = None
        self.initial_epoch = 0
        self.state = {}

        print(f" Checkpoint directory: {self.ckpt_dir}")

    # ── Internal helpers ──────────────────────────────────────

    def _load_state(self) -> dict:
        """Load training state from JSON if it exists."""
        # Clean up any abandoned .tmp file from a previous crash
        tmp = self.state_path.with_suffix('.tmp')
        if tmp.exists():
            tmp.unlink()
            print(" Cleaned up leftover .tmp state file")

        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                print(f" State loaded — last epoch: {state.get('last_epoch', 0)}, "
                      f"best {self.monitor}: {state.get('best_val_metric', 'N/A')}, "
                      f"patience counter: {state.get('patience_counter', 0)}")
                return state
            except (json.JSONDecodeError, OSError):
                print("️  State file corrupted — starting from last checkpoint only")
                return {}
        return {}

    def _get_latest_checkpoint(self):
        """Find the latest valid (non-corrupted) epoch checkpoint file."""
        pattern = str(self.ckpt_dir / f"{self.experiment_name}_epoch_*.keras")
        files = glob.glob(pattern)
        if not files:
            return None, 0

        def epoch_num(f):
            try:
                return int(f.split("_epoch_")[-1].split(".keras")[0])
            except Exception:
                return -1

        # Try from newest to oldest — skip suspiciously small (corrupted) files
        for f in sorted(files, key=epoch_num, reverse=True):
            if Path(f).stat().st_size > 1024:  # must be > 1KB to be valid
                return f, epoch_num(f)
            else:
                print(f" Checkpoint {Path(f).name} appears corrupted (too small) — trying previous...")

        print(" All checkpoints corrupted — starting from scratch")
        return None, 0

    def _build_callbacks(self) -> list:
        """Build all callbacks with restored state."""
        callbacks = []

        # 1. Best model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=str(self.best_model_path),
            monitor=self.monitor,
            save_best_only=True,
            mode=self.mode,
            verbose=1
        ))

        # 2. Per-epoch checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=self.epoch_template,
            save_freq=self.save_freq,
            save_best_only=False,
            verbose=0
        ))

        # 3. Stateful EarlyStopping (restores patience counter)
        callbacks.append(StatefulEarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            restore_best_weights=True,
            verbose=1,
            saved_best=self.state.get('best_val_metric', None),
            saved_patience_counter=self.state.get('patience_counter', 0)
        ))

        # 4. CSV Logger
        callbacks.append(CSVLogger(
            filename=str(self.csv_log_path),
            append=True
        ))

        # 5. Training state saver
        state_cb = TrainingStateCallback(
            state_path=self.state_path,
            monitor=self.monitor,
            mode=self.mode
        )
        state_cb.set_state(self.state)
        callbacks.append(state_cb)

        return callbacks

    def _check_already_complete(self) -> bool:
        """Return True if training was already marked complete."""
        if self.state.get('training_complete', False):
            print(" Training already complete! Nothing to resume.")
            return True
        return False

    # ── Public API ────────────────────────────────────────────

    def fit(self, train_data, val_data, epochs: int, **fit_kwargs):
        """
        Start or resume training.

        Args:
            train_data:  Training dataset or (x_train, y_train)
            val_data:    Validation dataset or (x_val, y_val)
            epochs (int): Total number of epochs (same value every session)
            **fit_kwargs: Any additional args passed to model.fit()

        Returns:
            Keras History object
        """

        # 1. Load state
        self.state = self._load_state()

        # 2. Check if already done
        if self._check_already_complete():
            return None

        # 3. Find latest checkpoint
        latest_ckpt, resume_epoch = self._get_latest_checkpoint()

        # 4. Load or build model
        if latest_ckpt:
            print(f"\n Resuming from epoch {resume_epoch} → {latest_ckpt}")
            self.model = tf.keras.models.load_model(latest_ckpt)
            self.initial_epoch = resume_epoch
        else:
            print("\n No checkpoint found — starting from scratch")
            self.model = self.model_fn()
            self.initial_epoch = 0

        # 5. Build callbacks
        callbacks = self._build_callbacks()

        # 6. Guard against wrong epochs value
        if self.initial_epoch >= epochs:
            print(f"  initial_epoch ({self.initial_epoch}) >= epochs ({epochs}). Nothing to train.")
            print("  Did you pass the same total epochs value as the original session?")
            return None

        # 7. Train
        fit_args = dict(
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=self.initial_epoch,   # <- critical
            callbacks=callbacks,
            **fit_kwargs
        )

        print(f"\n  Training from epoch {self.initial_epoch} → {epochs}\n")
        if isinstance(train_data, tf.data.Dataset):
            history = self.model.fit(train_data, **fit_args)
        elif isinstance(train_data, tuple):
            x, y = train_data
            history = self.model.fit(x, y, **fit_args)
        else:
            history = self.model.fit(train_data, **fit_args)

        return history

    def load_best_model(self):
        """Load and return the best saved model."""
        if self.best_model_path.exists():
            print(f" Loading best model from {self.best_model_path}")
            return tf.keras.models.load_model(str(self.best_model_path))
        else:
            raise FileNotFoundError(f"No best model found at {self.best_model_path}")

    def get_training_summary(self) -> dict:
        """Print and return the current training state."""
        state = self._load_state()
        print("\n── Training Summary ──────────────────────")
        for k, v in state.items():
            print(f"  {k}: {v}")
        print("──────────────────────────────────────────\n")
        return state


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────

"""
# ── In your Colab notebook ──────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

from resumable_trainer import ResumableTrainer

def create_model():
    model = tf.keras.Sequential([...])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

trainer = ResumableTrainer(
    project_name="Cifar_10",
    experiment_name="model_1",
    model_fn=create_model,
    checkpoint_root="/content/drive/MyDrive/Colab_Experiments",
    monitor="val_accuracy",
    mode="max",
    patience=7
)

history = trainer.fit(
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=64
)

# Load best model anytime
best_model = trainer.load_best_model()

# Check training progress
trainer.get_training_summary()
"""
