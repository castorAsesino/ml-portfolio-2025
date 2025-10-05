
"""
Script de ejemplo para entrenar y evaluar el LSTM con IMDB,
reutilizando tus utilidades existentes.
Ejecuta:
    python run_lstm_imdb.py
"""
import os
from lstm_data import IMDBData
from lstm_models import create_lstm_text_model
from lstm_train import TextTrainer, TrainConfig
# Importa tus utilidades existentes
from utils import save_training_results, save_summary_csv, plot_training_history, plot_confusion_matrix

def main():
    NOTEBOOK_NAME = "2_lstm_text_imdb"
    MODEL_NAME = "lstm_imdb_baseline"

    # 1) Datos
    data = IMDBData(num_words=20000, maxlen=200)
    (x_train, y_train), (x_test, y_test) = data.load_data()
    class_names = data.get_class_names()
    vocab_size = data.vocab_size()

    # 2) Modelo
    model = create_lstm_text_model(vocab_size=vocab_size, maxlen=200, embed_dim=128, lstm_units=128, bidirectional=True, dropout=0.3)
    trainer = TextTrainer(model, MODEL_NAME, TrainConfig(epochs=5, batch_size=128, lr=1e-3,
                                                         checkpoint_path=f"../results/{MODEL_NAME}_best.h5"))
    trainer.compile()

    # 3) Train
    history = trainer.train(x_train, y_train, x_test, y_test)  # IMDB ya viene split en train/test

    # 4) Eval
    test_loss, test_acc = trainer.evaluate(x_test, y_test)
    y_pred = trainer.predict_labels(x_test)
    f1 = trainer.f1_binary(y_test, y_pred)

    # 5) Persistir resultados en tu formato
    #    Tus utils requieren: history, test_accuracy, test_loss, f1, model_name, notebook_name, training_time, model_params
    results = save_training_results(
        history=history,
        test_accuracy=test_acc,
        test_loss=test_loss,
        f1=f1,
        model_name=MODEL_NAME,
        notebook_name=NOTEBOOK_NAME,
        training_time=trainer.get_training_time(),
        model_params=trainer.count_params()
    )
    save_summary_csv(results, NOTEBOOK_NAME)
    plot_training_history(history, NOTEBOOK_NAME, MODEL_NAME)

    # 6) Matriz de confusión
    #    utils.plot_confusion_matrix espera y_true/y_pred como índices de clase (0/1) y class_names
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        notebook_name=NOTEBOOK_NAME,
        model_name=MODEL_NAME
    )

    print("¡Listo! Revisa la carpeta results/.")

if __name__ == "__main__":
    main()
