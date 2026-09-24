"""Simple registry for tracking available model wrappers."""

from __future__ import annotations

import importlib

MODEL_REGISTRY: dict = {}

ALL_MODELS_MODULES = [
    "aims4pt.model_tools.Putirka_08",
    "aims4pt.model_tools.Neave_Putirka_17",
    "aims4pt.model_tools.Petrelli20",
    "aims4pt.model_tools.Higgins21",
    "aims4pt.model_tools.Jorgenson22",
    "aims4pt.model_tools.Agreda2024",
    "aims4pt.model_tools.Wang_21",
    "aims4pt.model_tools.Chicchi23",
    "aims4pt.model_tools.Masotta13",
]


def register_model(cls):
    """
    Decorator to register a model class.
    """
    name = cls.__name__
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_REGISTRY[name] = cls
    return cls


def initial_iterative_model(
    model_class,
    if_hydrous: bool,
    cpx_only: bool,
    T_P: str,
    iteration_max: int = 100,
    stop_criteria: float = 1e-2,
):
    '''
    Initialize a list of iterative model with given parameters.
    Parameters:
        model_class (class): The class of the model to be initialized.
        if_hydrous (bool): Whether to turn on the hydrous mode.
        cpx_only (bool): Whether the model must support cpx_only.
        T_P (str): "T" or "P"
        iteration_max (int): Maximum number of iterations.
        stop_criteria (float): Stopping criteria for the iteration.
    Returns:
        list: A list of initialized model instances.

    Notes:
    1. strictly chose if Cpx_only models.
    2. if_hydrous is True, then turn on the hydrous mode. Select models that support hydrous (hydrous or both). 
    3. if no models found, use anhydrous models.
    4. initial all possible combinations of T model and P model.
    '''
    final_model_list = []

    all_model_dict = model_class.cpx_only_models if cpx_only else model_class.cpx_liq_models
    if not all_model_dict:
        return final_model_list
    all_T_dict = all_model_dict.get("T_models", {})
    all_P_dict = all_model_dict.get("P_models", {})
    if not all_T_dict or not all_P_dict:
        # If no T or P models found, return empty list
        return final_model_list

    # Helper to filter models by hydrous support
    def filter_models(model_dict, if_hydrous: bool):
        if not model_dict:
            return []
        filtered = []
        if not if_hydrous:
            filtered = model_dict.get("anhydrous", []) + model_dict.get("both", [])
        else:
            filtered = model_dict.get("hydrous", []) + model_dict.get("both", [])
            if not filtered:
                filtered = model_dict.get("anhydrous", [])
        return filtered

    T_model_lists = filter_models(all_T_dict, if_hydrous)
    P_model_lists = filter_models(all_P_dict, if_hydrous)

    if not T_model_lists or not P_model_lists:
        # If no models found, return empty list
        return final_model_list
    
    # initialize all combinations of T and P models
    for T_model in T_model_lists:
        for P_model in P_model_lists:
            try:
                model = model_class(T_P=T_P, T_model_name=T_model, P_model_name=P_model, 
                                    iteration_max=iteration_max, stop_criteria=stop_criteria)
                final_model_list.append(model)
            except Exception as e:
                print(f"[skipped] Error initializing model {model_class} with T: {T_model}, P: {P_model} - {e}")

    return final_model_list


def get_models_initial_pools(T_P: str = "T", method_type: str = "cpx_only", if_hydrous: bool = True):
    """
    Return a list of model instances that match the given T_P and cpx_only.

    Parameters:
        T_P (str): "T" or "P"
        method_type (str): "cpx_only" or "cpx_liq" or "both".
        if_hydrous (bool): Whether turn on the hydrous mode (only include models that support hydrous or both).

    Returns:
        list[model instance]
    """
    if method_type == "cpx_only":
        cpx_only_list = [True] # only return cpx_only models
    elif method_type == "cpx_liq":
        cpx_only_list = [False]  # only return cpx_liq models
    elif method_type == "both":
        cpx_only_list = [True, False] # return both cpx_only and cpx_liq models

    result = []
    for cpx_only in cpx_only_list:
        for name, model_class in MODEL_REGISTRY.items():
            try:
                # check if iterative model
                init_args = model_class.__init__.__code__.co_varnames
                # print(getattr(model_class, "iterative_model", False))
                if getattr(model_class, "iterative_model", False):
                    model_list = initial_iterative_model(model_class=model_class, T_P=T_P, if_hydrous=if_hydrous, cpx_only=cpx_only,  )
                    if model_list:
                        result.extend(model_list)
                    continue
                if "cpx_only" in init_args:
                    model = model_class(T_P=T_P, cpx_only=cpx_only)
                else:
                    model = model_class(T_P=T_P)

                if getattr(model, "cpx_only", False) == cpx_only: #
                    result.append(model)

            except Exception as e:
                print(f"[Skip] Error initializing model {name}: {e}")
    return result


def print_all_registered_thermobarometry():
    """Print all registered models."""
    for model_name in MODEL_REGISTRY:
        print(model_name)


def import_all_models(skip_failed: bool = True):
    """
    Import all registered model modules.

    Parameters:
        skip_failed (bool): If True, keep importing after a module fails.

    Returns:
        tuple[list[str], dict[str, str]]:
            Imported module names and a mapping of failed modules to errors.
    """
    imported = []
    failed = {}

    for module in ALL_MODELS_MODULES:
        try:
            importlib.import_module(module)
            imported.append(module)
        except Exception as exc:
            if not skip_failed:
                raise
            failed[module] = str(exc)

    return imported, failed

if __name__ == "__main__":
    print_all_registered_thermobarometry()
