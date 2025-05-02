import optuna
import yaml
import torch
from pathlib import Path
from .config import Config
from .config_trainer import ConfigTrainer
import csv

def log_trial_results(trial, val_rmse):
    log_path = "optuna_runs/testing_env.csv"
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "val_rmse"] + list(trial.params.keys()))
        
        if f.tell() == 0:  # Write header if file is empty
            writer.writeheader()
        
        # prep trial results
        trial_result = {"trial_id": trial.number, "val_rmse": val_rmse}
        trial_result.update(trial.params)
        
        # write results
        writer.writerow(trial_result)


def objective(trial):

    print("Before model initialization:")
    print(torch.cuda.memory_summary())

    torch.cuda.empty_cache()

    # hyperparameters range
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    depth = trial.suggest_int("depth", 2, 14)

    width = trial.suggest_categorical("width", [16, 32, 64, 128])

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax"])

    # load base YAML config
    with open("configs/optuna_template.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # update YAML config with suggested hyperparameters
    config_dict["OPTIMIZER"]["NAME"] = optimizer
    config_dict["OPTIMIZER"]["LEARNING_RATE"] = learning_rate
    config_dict["OPTIMIZER"]["WEIGHT_DECAY"] = weight_decay
    config_dict["OPTIMIZER"]["BATCH_SIZE"] = batch_size
    config_dict["ARCHITECTURE"]["DEPTH"] = depth
    config_dict["ARCHITECTURE"]["WIDTH"] = width

    # save updated config to a temporary file
    temp_config_path = "configs/updated_optuna_template.yaml"
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(config_dict, f)

    # use updated YAML config
    config = Config(temp_config_path)
    config_trainer = ConfigTrainer(
        config,
        Path("testing_env"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=4 if torch.cuda.is_available() else 1, 
    )

    # set hyperparameters before fitting model
    hyperparams = config.get_module_params()
    config_trainer.set_hyperparams(hyperparams)

    # Retrieve and set the model
    model = config.get_model(config_trainer.input_dim)

    # Train the model
    config_trainer.fit(model)

    # Retrieve validation RMSE normalized by b
    val_rmse = config_trainer.trainer.callback_metrics.get("residual/val_rmse_normalized_by_b")
    val_rmse = val_rmse.item()
    log_trial_results(trial, val_rmse)
    print(f"Validation RMSE normalized by b: {val_rmse}")

    print("After model initialization:")
    print(torch.cuda.memory_summary())

    torch.cuda.empty_cache()

    print("After emptying cache initialization:")
    print(torch.cuda.memory_summary)

    return val_rmse


def run_optuna_study():
    study_name = "cfd_tuning"
    storage = "sqlite:///optuna_study.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage, 
        load_if_exists=True,  
    )
    study.optimize(objective, n_trials=1000)  

    # print & save the best hyperparameters
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    with open("best_hyperparameters.yaml", "w") as f:
        yaml.safe_dump(study.best_trial.params, f)
