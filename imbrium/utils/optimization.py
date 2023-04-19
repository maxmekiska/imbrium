import warnings

import optuna

warnings.filterwarnings("ignore", category=UserWarning)


def seeker(
    optimizer_range,
    layer_config_range,
    optimizer_args_range,
    optimization_target,
    n_trials,
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def objective(trial):
                warnings.filterwarnings("ignore", category=UserWarning)
                param1 = trial.suggest_categorical("optimizer", optimizer_range)
                param2 = trial.suggest_categorical("layer_config", layer_config_range)
                param3 = trial.suggest_categorical(
                    "optimizer_args", optimizer_args_range
                )
                try:
                    result = func(
                        *args,
                        optimizer=param1,
                        layer_config=param2,
                        optimizer_args=param3,
                        **kwargs,
                    )
                    return result
                except Exception as e:
                    # Print the error message and return infinity
                    print(f"Error during trial {trial.number}: {e}")
                    if optimization_target == "minimize":
                        return float("inf")
                    elif optimization_target == "maximize":
                        return -float("inf")
                    else:
                        return None

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
