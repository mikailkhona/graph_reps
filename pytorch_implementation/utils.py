import os
import sys
import torch
import numpy as np
import random


def initialize_and_log(deploy=False, tag='scratch', wandb_project_name='project'):
    """
    Initializes wandb if available and requested, and sets up logging.
    
    Parameters:
    - deploy: Boolean indicating whether to deploy with wandb.
    - tag: Tag or name for the log directory.
    - wandb_project_name: Name of the wandb project for initialization.
    """
    log_directory = './logs/' + tag
    os.makedirs(log_directory, exist_ok=True)
    
    if deploy :
        try:
            import wandb
            # Attempt to initialize wandb if not already done
            if wandb.run is None:
                wandb.init(project=wandb_project_name)
                print(f"Initialized wandb project: {wandb_project_name}")
                wandb.run.name = wandb.run.id
                wandb.run.save()
            # Use the wandb run ID to name the log file
            fname = os.path.join(log_directory, wandb.run.id + ".log")
        except ImportError:
            print("wandb is not installed, proceeding without wandb integration.")
            fname = os.path.join(log_directory, "default.log")
    else:
        # If not deploying with wandb, use a default log file name
        fname = os.path.join(log_directory, "default.log")
    
    # Open the log file and redirect stdout and stderr
    fout = open(fname, "a", 1)
    sys.stdout = fout
    sys.stderr = fout
    return fout


def cleanup(deploy, fp):
    if deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()

def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)