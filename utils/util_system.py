import os
import yaml
import wandb
import torch
import onnx
import inspect

from loguru import logger
from utils.decorators import *


@logger_wraps()
def parse_yaml(path):
    """
    Parse and return the contents of a YAML file.

    Args:
        path (str): Path to the YAML file to be parsed.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the provided path does not point to an existing file.
    """
    try:
        with open(path, 'r') as yaml_file:
            config_dict = yaml.full_load(yaml_file)
        return config_dict
    except FileNotFoundError:
        raise

@logger_wraps()
def wandb_setup(yaml_dict):
    """
    Configures and initializes a Weights & Biases (wandb) run using the parameters provided in the YAML configuration.

    This function sets up logging, tracking, and visualization capabilities for a machine learning experiment
    using wandb. The function reads the wandb-specific configuration from the provided YAML dictionary and
    initializes a new wandb run with those settings.

    Args:
        yaml_dict (dict): Dictionary containing the wandb configuration loaded from a YAML file.

    Returns:
        wandb.wandb_run.Run: An instance of a wandb run, which provides methods and properties to 
            interact with the experiment being tracked.

    Example:
        >>> config = parse_yaml("config.yaml")
        >>> run = wandb_setup(config)
    """
    api_key=yaml_dict['wandb']['login']['key']
    if api_key == "":
        logger.warning("WandB login key is empty, aborting setup.")
        return None
    try:
        wandb.login(key=api_key)
        run_config = {k: v for k, v in yaml_dict['wandb']['init'].items()}
        run = wandb.init(**run_config)
        return run
    except Exception as e:
        logger.error(f"WandB setup failed: {e}")
        raise


def log_model_information_to_wandb(wandb_run, model, input_shape, root_path):
    """
    Logs the architecture and ONNX file of the given model to Weights & Biases (wandb) as artifacts.
    # Ref: https://docs.wandb.ai/ref/python/artifact

    This function creates a new wandb artifact of type "model" to store the architecture
    of the provided PyTorch model. The architecture details, including the names and types of all modules
    in the model, are written to a file named "model_arch.txt". This file is then attached to the created artifact.
    Finally, the artifact is logged to the provided wandb run.

    Args:
        wandb_run (wandb.wandb_run.Run): The current wandb run.
        model (torch.nn.Module): The PyTorch model.
        input_shape (tuple): The shape of the input tensor required by the model for ONNX conversion.
        root_path (str): Path to the directory to log.

    Example:
        >>> wandb_run = wandb.init()
        >>> model = SomePyTorchModel()
        >>> log_model_architecture_to_wandb(wandb_run, model)
    """
    if not wandb_run:
        logger.error("Invalid wandb_run provided.")
        return
    
    try: # Log model architecture
        artifact_arch = wandb.Artifact("architecture", type="model", description="Architecture of the trained model", metadata={"framework": "pytorch"})
        with artifact_arch.new_file("model_arch.txt", mode="w") as f:
            for name, module in model.named_modules(): f.write(f"{name}: {type(module).__name__}\n")
        wandb_run.log_artifact(artifact_arch)
    except Exception as e: logger.error(f"Error in logging model architecture: {e}")
    
    try: # Convert and log model to ONNX
        onnx_file_name = "model.onnx"
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(model, dummy_input, onnx_file_name)
        artifact_onnx = wandb.Artifact("architecture", type="model", description="ONNX model file")
        artifact_onnx.add_file(onnx_file_name)
        wandb_run.log_artifact(artifact_onnx)
        logger.info(f"ONNX model saved and logged to wandb: {onnx_file_name}")
    except Exception as e: logger.error(f"Failed to save and log ONNX model: {e}")
    
    # try: # Log directory files
    #     def add_files_to_artifact(dir_path, artifact):
    #         for item in os.listdir(dir_path):
    #             item_path = os.path.join(dir_path, item)
    #             if os.path.isfile(item_path): artifact.add_file(item_path, name=item_path[len(dir_path)+1:])
    #             elif os.path.isdir(item_path): add_files_to_artifact(item_path, artifact)
    #     artifact_dirfiles = wandb.Artifact("files", type="data", description="All files and subdirectories from the specified directory")
    #     add_files_to_artifact(root_path, artifact_dirfiles)
    #     wandb_run.log_artifact(artifact_dirfiles)
    # except Exception as e: logger.error(f"Error in logging directory files: {e}")
    
    logger.debug(f"Complete {__name__}.{inspect.currentframe().f_code.co_name}")