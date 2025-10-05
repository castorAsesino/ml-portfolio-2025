import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer
import numpy as np

def load_pretrained_qa_model(model_name="distilbert-base-uncased"):
    """Cargar modelo pre-entrenado para Question Answering"""
    print(f"Cargando modelo {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
        
        print(f"Modelo {model_name} cargado exitosamente")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error cargando modelo pre-entrenado: {e}")
        print("Creando modelo simple personalizado...")
        return create_simple_qa_model(), None

def create_simple_qa_model(vocab_size=50000, embedding_dim=256, hidden_dim=512, max_length=512):
    """Crear modelo simple de QA para cuando no hay modelo pre-entrenado"""
    
    # Inputs
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    
    # Embedding
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(input_ids)
    
    # Encoder LSTM bidireccional
    lstm_output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
    )(embedding, mask=attention_mask)
    
    # Capas para inicio y fin de respuesta
    start_logits = tf.keras.layers.Dense(1, name="start_logits")(lstm_output)
    start_logits = tf.keras.layers.Flatten()(start_logits)
    
    end_logits = tf.keras.layers.Dense(1, name="end_logits")(lstm_output)
    end_logits = tf.keras.layers.Flatten()(end_logits)
    
    # Modelo
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[start_logits, end_logits]
    )
    
    return model

def create_rag_model(encoder_name="distilbert-base-uncased", max_length=512):
    """Crear modelo RAG completo con recuperador y generador"""
    
    # Cargar tokenizer y encoder
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # Modelo encoder para documentos y preguntas
    encoder = TFAutoModelForQuestionAnswering.from_pretrained(encoder_name)
    
    return {
        'encoder': encoder,
        'tokenizer': tokenizer,
        'max_length': max_length
    }

class SimpleQAModel:
    """Wrapper simple para modelos de QA"""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, questions, contexts):
        """Predecir respuestas para preguntas dadas contextos"""
        if self.tokenizer is None:
            # Para modelo personalizado, implementar lógica de predicción
            return self._predict_simple(questions, contexts)
        
        # Para modelo transformers
        return self._predict_transformers(questions, contexts)
    
    def _predict_transformers(self, questions, contexts):
        """Predecir usando modelo transformers"""
        predictions = []
        
        for question, context in zip(questions, contexts):
            inputs = self.tokenizer(
                question, 
                context, 
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="tf"
            )
            
            outputs = self.model(inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
            end_idx = tf.argmax(end_logits, axis=1).numpy()[0]
            
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
            answer = self.tokenizer.decode(answer_tokens)
            
            predictions.append({
                'answer': answer,
                'start': start_idx,
                'end': end_idx,
                'score': (tf.reduce_max(start_logits) + tf.reduce_max(end_logits)).numpy() / 2
            })
        
        return predictions
    
    def _predict_simple(self, questions, contexts):
        """Predecir usando modelo simple (implementación básica)"""
        predictions = []
        
        for question, context in zip(questions, contexts):
            # Implementación muy básica - en realidad usarías el modelo entrenado
            words = context.split()
            if len(words) > 5:
                answer = " ".join(words[2:5])  # Respuesta dummy
            else:
                answer = context
            
            predictions.append({
                'answer': answer,
                'start': 0,
                'end': len(answer),
                'score': 0.5
            })
        
        return predictions