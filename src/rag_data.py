import json
import pandas as pd
from datasets import load_dataset
import tensorflow as tf
import os

class QADataLoader:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_squad_data(self, version="squad_v2"):
        """Cargar dataset SQuAD para Question Answering"""
        print("Cargando dataset SQuAD...")
        
        try:
            # Cargar dataset SQuAD
            dataset = load_dataset("squad_v2" if version == "squad_v2" else "squad")
            
            # Preparar datos en formato simple
            train_data = self._prepare_qa_data(dataset['train'])
            val_data = self._prepare_qa_data(dataset['validation'])
            
            print(f"SQuAD cargado: {len(train_data)} ejemplos entrenamiento, {len(val_data)} validación")
            return train_data, val_data, val_data  # Usamos validation como test también
            
        except Exception as e:
            print(f"Error cargando SQuAD: {e}")
            print("Creando dataset de ejemplo...")
            return self._create_sample_qa_data()
    
    def _prepare_qa_data(self, dataset):
        """Preparar datos en formato simple para QA"""
        processed_data = []
        for item in dataset:
            # Para SQuAD v2, algunas preguntas no tienen respuesta
            if len(item['answers']['text']) > 0:
                processed_data.append({
                    'context': item['context'],
                    'question': item['question'],
                    'answer': item['answers']['text'][0],
                    'answer_start': item['answers']['answer_start'][0]
                })
        return processed_data
    
    def _create_sample_qa_data(self):
        """Crear dataset de ejemplo si no se puede cargar SQuAD"""
        print("Creando dataset de ejemplo para QA...")
        
        sample_data = [
            {
                'context': 'El machine learning es un subcampo de la inteligencia artificial que se centra en el desarrollo de algoritmos que pueden aprender de los datos y hacer predicciones.',
                'question': '¿Qué es el machine learning?',
                'answer': 'un subcampo de la inteligencia artificial',
                'answer_start': 3
            },
            {
                'context': 'Las redes neuronales convolucionales (CNN) son especialmente efectivas para el procesamiento de imágenes y reconocimiento de patrones visuales.',
                'question': '¿Para qué son efectivas las CNN?',
                'answer': 'para el procesamiento de imágenes y reconocimiento de patrones visuales',
                'answer_start': 56
            },
            {
                'context': 'Python es un lenguaje de programación interpretado de alto nivel y de propósito general. Fue creado por Guido van Rossum en 1991.',
                'question': '¿Quién creó Python?',
                'answer': 'Guido van Rossum',
                'answer_start': 89
            },
            {
                'context': 'El deep learning utiliza redes neuronales con múltiples capas para aprender representaciones de datos complejos.',
                'question': '¿Qué utiliza el deep learning?',
                'answer': 'redes neuronales con múltiples capas',
                'answer_start': 22
            },
            {
                'context': 'TensorFlow y PyTorch son dos de los frameworks más populares para el desarrollo de modelos de machine learning.',
                'question': '¿Cuáles son frameworks populares para ML?',
                'answer': 'TensorFlow y PyTorch',
                'answer_start': 0
            }
        ]
        
        # Dividir en train/val/test
        train_data = sample_data[:3]
        val_data = sample_data[3:4]
        test_data = sample_data[4:]
        
        return train_data, val_data, test_data
    
    def load_custom_qa_data(self, file_path):
        """Cargar datos de QA personalizados desde archivo"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            data = pd.read_csv(file_path).to_dict('records')
        else:
            raise ValueError("Formato de archivo no soportado")
        
        return self._split_qa_data(data)
    
    def _split_qa_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """Dividir datos en train/val/test"""
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        return train_data, val_data, test_data

class DocumentRetriever:
    """Clase simple para recuperación de documentos (para RAG)"""
    
    def __init__(self, documents=None):
        self.documents = documents or []
    
    def add_documents(self, documents):
        """Añadir documentos a la colección"""
        self.documents.extend(documents)
    
    def retrieve(self, query, top_k=3):
        """Recuperar documentos más relevantes para una consulta"""
        # Implementación simple basada en coincidencia de términos
        # En una implementación real usarías embeddings + similitud coseno
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            if isinstance(doc, dict):
                doc_text = doc.get('text', doc.get('context', ''))
            else:
                doc_text = str(doc)
            
            doc_terms = set(doc_text.lower().split())
            score = len(query_terms.intersection(doc_terms))
            scored_docs.append((doc, score))
        
        # Ordenar por relevancia y devolver top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]