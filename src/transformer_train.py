
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

@dataclass
class HfTrainConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "results/transformers_imdb"
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 256
    eval_strategy: str = "epoch"      # "epoch" para tener métricas por época
    save_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True

class HistoryCollector(TrainerCallback):
    """
    Colecciona métricas por época para construir un "history" similar a Keras.
    """
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Solo guardamos si hay valores, el Trainer loguea frecuentemente
            self.logs.append({k: float(v) for k, v in logs.items() if isinstance(v, (int, float))})

    def to_history(self) -> Dict[str, list]:
        hist = {}
        for entry in self.logs:
            for k, v in entry.items():
                hist.setdefault(k, []).append(v)
        # Convertimos algunas claves comunes al formato esperado por curvas:
        history = {}
        # loss durante entrenamiento
        if "loss" in hist:
            history["loss"] = hist["loss"]
        # evaluación por época
        if "eval_loss" in hist:
            history["val_loss"] = hist["eval_loss"]
        if "eval_accuracy" in hist:
            history["val_accuracy"] = hist["eval_accuracy"]
        if "eval_f1" in hist:
            history["val_f1"] = hist["eval_f1"]
        return history

class SimpleHistory:
    def __init__(self, history_dict):
        self.history = history_dict

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

def train_and_eval(model, tokenizer, tokenized, cfg: HfTrainConfig):
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        evaluation_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        logging_strategy=cfg.logging_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to="none"
    )

    collector = HistoryCollector()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[collector]
    )

    t0 = time.time()
    trainer.train()
    training_time = time.time() - t0

    # Evaluación final en test
    test_metrics = trainer.evaluate(tokenized["test"])
    # Predicciones para matriz de confusión
    preds_out = trainer.predict(tokenized["test"])
    y_true = preds_out.label_ids
    y_pred = np.argmax(preds_out.predictions, axis=-1)

    # Construir history Keras-like
    history = SimpleHistory(collector.to_history())

    return trainer, history, test_metrics, y_true, y_pred, training_time
