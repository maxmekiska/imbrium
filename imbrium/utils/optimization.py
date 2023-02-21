import warnings

import optuna

warnings.filterwarnings("ignore", category=UserWarning)


def seeker(optimizer_range, layer_config_range, optimization_target, n_trials):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def objective(trial):
                warnings.filterwarnings("ignore", category=UserWarning)
                param1 = trial.suggest_categorical("optimizer", optimizer_range)
                param2 = trial.suggest_categorical("layer_config", layer_config_range)

                result = func(*args, optimizer=param1, layer_config=param2, **kwargs)
                return result

            study = optuna.create_study(direction=optimization_target)
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            best_result = study.best_value
            print(f"Best hyperparameters: {best_params}")
            print(f"Best result: {best_result}")

            result = func(*args, **best_params, **kwargs)
            return result

        return wrapper

    return decorator
