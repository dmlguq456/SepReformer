import os
import torch
from loguru import logger
from torchinfo import summary as summary_
from ptflops import get_model_complexity_info
from thop import profile
import numpy as np
import torch



def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer, location):
    """
    Load the latest checkpoint (model state and optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model into which the checkpoint's model state should be loaded.
        optimizer (torch.optim.Optimizer): The optimizer into which the checkpoint's optimizer state should be loaded.
        location (str, optional): Device location for loading the checkpoint. Defaults to 'cpu'.

    Returns:
        int: The epoch number associated with the loaded checkpoint. 
             If no checkpoint is found, returns 0 as the starting epoch.

    Notes:
        - The checkpoint file is expected to have keys: 'model_state_dict', 'optimizer_state_dict', and 'epoch'.
        - If there are multiple checkpoint files in the directory, the one with the highest epoch number is loaded.
    """
    # List all .pkl files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)]

    # If there are no checkpoint files, return 0 as the starting epoch
    if not checkpoint_files: return 1
    else:
        # Extract the epoch numbers from the file names and find the latest (max)
        epochs = [int(f.split('.')[1]) for f in checkpoint_files]
        latest_checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[epochs.index(max(epochs))])

        # Load the checkpoint into the model & optimizer
        logger.info(f"Loaded Pretrained model from {latest_checkpoint_file} .....")
        checkpoint_dict = torch.load(latest_checkpoint_file, map_location=location)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False) # Depend on weight file's key!!
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        # Retrun latent epoch
        return checkpoint_dict['epoch'] + 1
    
def save_checkpoint_per_nth(nth, epoch, model, optimizer, train_loss, valid_loss, checkpoint_path, wandb_run):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if epoch % nth == 0:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # Log and save the checkpoint file using wandb
        wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))

def save_checkpoint_per_best(best, valid_loss, train_loss, epoch, model, optimizer, checkpoint_path):
    """
    Save the state of the model and optimizer every nth epoch to a checkpoint file.
    Additionally, log and save the checkpoint file using wandb.

    Args:
        nth (int): Interval for which checkpoints should be saved.
        epoch (int): The current training epoch.
        model (nn.Module): The model whose state needs to be saved.
        optimizer (Optimizer): The optimizer whose state needs to be saved.
        checkpoint_path (str): Directory path where the checkpoint will be saved.
        wandb_run (wandb.wandb_run.Run): The current wandb run to log and save the checkpoint.

    Returns:
        None
    """
    if valid_loss < best:
        # Save the state of the model and optimizer to a checkpoint file
        torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss
                    },
                    os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        
        # # Log and save the checkpoint file using wandb
        # wandb_run.save(os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth"))
        best = valid_loss
    return best

def step_scheduler(scheduler, **kwargs):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    # Add another schedulers
    else:
        raise ValueError(f"Unknown scheduler type: {type(scheduler)}")

def print_parameters_count(model):
    total_parameters = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_parameters += param_count
        logger.info(f"{name}: {param_count}")
    logger.info(f"Total parameters: {(total_parameters / 1e6):.2f}M")

def model_params_mac_summary(model, input, dummy_input, metrics):
    
    # ptflpos
    if 'ptflops' in metrics:
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, (input.shape[1],), print_per_layer_stat=False, verbose=False) # (num_samples,)
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        MACs_thop, params_thop = profile(model, inputs=(input, ), verbose=False)
        MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=input.size(), verbose=0)
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e6, model_profile.total_params/1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")
