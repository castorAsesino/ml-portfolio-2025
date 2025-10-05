
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def create_transformer(model_name="distilbert-base-uncased", num_labels=2):
    """
    Retorna (tokenizer, model) para clasificación de secuencias.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model
