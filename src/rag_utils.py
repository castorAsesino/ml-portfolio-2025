import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score

def evaluate_qa_model(model, tokenizer, test_data, notebook_name):
    """Evaluar modelo de QA y guardar resultados"""
    print("Evaluando modelo de QA...")
    
    # Preparar datos para evaluación
    questions = [item['question'] for item in test_data]
    contexts = [item['context'] for item in test_data]
    true_answers = [item['answer'] for item in test_data]
    
    # Hacer predicciones
    if tokenizer is not None:
        predictions = _predict_with_transformers(model, tokenizer, questions, contexts)
    else:
        predictions = _predict_with_custom_model(model, questions, contexts)
    
    # Calcular métricas
    exact_match = calculate_exact_match(predictions, true_answers)
    f1 = calculate_f1_score(predictions, true_answers)
    
    # Guardar predicciones
    results_dir = f"../results/{notebook_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Crear dataframe con resultados
    results_df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'true_answer': true_answers,
        'predicted_answer': [p['answer'] for p in predictions],
        'score': [p['score'] for p in predictions],
        'is_correct': [is_answer_correct(p['answer'], true) for p, true in zip(predictions, true_answers)]
    })
    
    # Guardar resultados detallados
    results_path = f"{results_dir}/qa_predictions.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    
    metrics = {
        'exact_match': exact_match,
        'f1': f1,
        'loss': 0.45,  # Simulado para ejemplo
        'total_samples': len(test_data),
        'correct_predictions': results_df['is_correct'].sum()
    }
    
    # Guardar métricas
    metrics_path = f"{results_dir}/qa_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados guardados en: {results_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    
    return metrics, results_df.to_dict('records')

def _predict_with_transformers(model, tokenizer, questions, contexts):
    """Hacer predicciones con modelo transformers"""
    predictions = []
    
    for question, context in zip(questions, contexts):
        try:
            inputs = tokenizer(
                question, 
                context, 
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="tf"
            )
            
            outputs = model(inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
            end_idx = tf.argmax(end_logits, axis=1).numpy()[0]
            
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            score = (tf.reduce_max(start_logits) + tf.reduce_max(end_logits)).numpy() / 2
            
            predictions.append({
                'answer': answer,
                'start': start_idx.numpy() if hasattr(start_idx, 'numpy') else start_idx,
                'end': end_idx.numpy() if hasattr(end_idx, 'numpy') else end_idx,
                'score': float(score)
            })
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            predictions.append({
                'answer': 'Error en predicción',
                'start': 0,
                'end': 0,
                'score': 0.0
            })
    
    return predictions

def _predict_with_custom_model(model, questions, contexts):
    """Hacer predicciones con modelo personalizado"""
    predictions = []
    
    for question, context in zip(questions, contexts):
        # Implementación básica para modelo personalizado
        words = context.split()
        if len(words) > 3:
            answer = " ".join(words[1:3])  # Respuesta simple
        else:
            answer = context
        
        predictions.append({
            'answer': answer,
            'start': 0,
            'end': len(answer),
            'score': 0.7
        })
    
    return predictions

def calculate_exact_match(predictions, true_answers):
    """Calcular exact match entre predicciones y respuestas verdaderas"""
    correct = 0
    for pred, true in zip(predictions, true_answers):
        if is_answer_correct(pred['answer'], true):
            correct += 1
    return correct / len(predictions)

def calculate_f1_score(predictions, true_answers):
    """Calcular F1 score promedio"""
    f1_scores = []
    for pred, true in zip(predictions, true_answers):
        pred_tokens = set(pred['answer'].lower().split())
        true_tokens = set(true.lower().split())
        
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        common_tokens = pred_tokens.intersection(true_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return np.mean(f1_scores)

def is_answer_correct(predicted, true):
    """Verificar si la respuesta predicha es correcta"""
    return predicted.strip().lower() == true.strip().lower()

def plot_qa_examples(predictions, test_data, notebook_name, num_examples=5):
    """Visualizar ejemplos de preguntas y respuestas"""
    results_dir = f"../results/{notebook_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(predictions), min(num_examples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        pred = predictions[idx]
        true_data = test_data[idx]
        
        context = true_data['context']
        question = true_data['question']
        true_answer = true_data['answer']
        predicted_answer = pred['answer']
        is_correct = is_answer_correct(predicted_answer, true_answer)
        
        # Truncar contexto si es muy largo
        if len(context) > 300:
            context = context[:300] + "..."
        
        # Crear visualización
        axes[i].axis('off')
        axes[i].text(0.02, 0.95, f"Contexto: {context}", 
                    transform=axes[i].transAxes, fontsize=10, wrap=True,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        axes[i].text(0.02, 0.65, f"Pregunta: {question}", 
                    transform=axes[i].transAxes, fontsize=11, weight='bold', wrap=True,
                    verticalalignment='top')
        
        color = "lightgreen" if is_correct else "lightcoral"
        axes[i].text(0.02, 0.45, f"Respuesta verdadera: {true_answer}", 
                    transform=axes[i].transAxes, fontsize=10, wrap=True,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        axes[i].text(0.02, 0.25, f"Respuesta predicha: {predicted_answer}", 
                    transform=axes[i].transAxes, fontsize=10, wrap=True,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        status = "✓ CORRECTO" if is_correct else "✗ INCORRECTO"
        axes[i].text(0.02, 0.05, f"Estado: {status} (Score: {pred['score']:.3f})", 
                    transform=axes[i].transAxes, fontsize=12, weight='bold',
                    verticalalignment='top', color="green" if is_correct else "red")
    
    plt.tight_layout()
    plot_path = f"{results_dir}/qa_examples.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ejemplos de QA guardados en: {plot_path}")

def save_qa_results(history, metrics, predictions, model_name, notebook_name, training_time, model_params):
    """Guardar resultados completos de QA"""
    results_dir = f"../results/{notebook_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Preparar resultados
    results = {
        'model_name': model_name,
        'notebook_name': notebook_name,
        'metrics': metrics,
        'training_time': training_time,
        'model_params': model_params,
        'num_predictions': len(predictions),
        'history': history.history if history else {}
    }
    
    # Guardar en JSON
    results_path = f"{results_dir}/training_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados de entrenamiento guardados en: {results_path}")
    return results