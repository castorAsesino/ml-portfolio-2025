
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_imdb_splits(test_size_val=0.1, seed=42):
    """
    Carga IMDB desde Hugging Face Datasets y separa train/val/test.
    Devuelve (raw_datasets, class_names)
    - raw_datasets['train'], raw_datasets['validation'], raw_datasets['test']
    Cada elemento es un Dataset con columnas: 'text', 'label'
    """
    ds = load_dataset("imdb")
    # Crear validación a partir de train
    train_ds = ds["train"]
    train_idx, val_idx = train_test_split(range(len(train_ds)), test_size=test_size_val, random_state=seed, shuffle=True, stratify=train_ds["label"])
    raw_train = train_ds.select(train_idx)
    raw_val   = train_ds.select(val_idx)
    raw_test  = ds["test"]
    class_names = ["negativo", "positivo"]
    return {"train": raw_train, "validation": raw_val, "test": raw_test}, class_names

def tokenize_function(examples, tokenizer, text_field="text", max_length=256):
    return tokenizer(examples[text_field], truncation=True, padding="max_length", max_length=max_length)

def prepare_tokenized_datasets(raw_datasets, tokenizer, max_length=256):
    tokenized = {}
    for split in ["train", "validation", "test"]:
        tokenized[split] = raw_datasets[split].map(
            lambda x: tokenize_function(x, tokenizer, max_length=max_length),
            batched=True,
            remove_columns=[c for c in raw_datasets[split].column_names if c not in ["label"]]
        )
        tokenized[split].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized
