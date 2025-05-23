"""
main.py

Main training script that orchestrates dataset loading, model building,
training, evaluation, and saving results using parameters defined in config.ini.

Modules used:
- config: Load training configuration from config.ini file
- data_loader: Create train and test datasets
- model_builder: Build and compile models based on model names
- callbacks: Setup early stopping and learning rate scheduling callbacks
- evaluation: Evaluate the trained model and save accuracy/loss results
- set_seed: Set random seed for reproducibility
"""

from config import load_config
from data_loader import create_train_test_datasets
from model_builder import create_model, compile_model
from callbacks import get_callbacks
from evaluation import evaluate_and_save_results
from set_seed import set_seed
import os
import tensorflow as tf

if __name__ == '__main__':
    config = load_config(config_path="../config/config.ini")

    seed = int(config["RANDOM"]["seed"])
    dataset_dir = config["DATA"]["dataset_dir"]
    result_dir = config["RESULTS"]["result_dir"]
    dataset_name = config["MODEL"]["dataset_name"]
    img_size = tuple(map(int, config["MODEL"]["img_size"].split(',')))
    batch_size = int(config["MODEL"]["batch_size"])
    epochs = int(config["MODEL"]["epochs"])
    class_num = int(config["MODEL"]["class_num"])
    model_names = config["MODEL"]["model_names"].split(',')
    patience_es = int(config["MODEL"]["patience_es"])
    patience_lr = int(config["MODEL"]["patience_lr"])

    set_seed(seed=seed)

    acc_loss = {}

    for model_name in model_names:
        res_dir = os.path.join(result_dir, f"{dataset_name}-LastTrain-{model_name}")
        os.makedirs(res_dir, exist_ok=True)

        print(f"{model_name} için eğitime başlandı.")

        train_data, test_data = create_train_test_datasets(
            data_dir=dataset_dir,
            img_size=img_size,
            batch_size=batch_size
        )

        model = create_model(
            class_num=int(class_num),
            input_shape=(img_size[0], img_size[1], 3),
            model_name=model_name
        )
        model = compile_model(model)

        early_stop, reduce_lr = get_callbacks(patience_es=patience_es, patience_lr=patience_lr)

        model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            shuffle=True,
            verbose=1,
            callbacks=[early_stop, reduce_lr]
        )

        model_path = os.path.join(res_dir, f"{dataset_name}{model_name}_adam_1e4")
        tf.saved_model.save(model, model_path)

        results = evaluate_and_save_results(model, test_data, res_dir, model_name, dataset_name)
        acc_loss[model_name] = results

