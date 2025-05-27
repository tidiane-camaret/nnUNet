import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.multi_channel_transform import AddPromptChannelsTransform
from typing import Union, Tuple, List
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
import numpy as np
from torch import nn
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights # Import the function

class CustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Ensure all original args for nnUNetTrainer are passed, especially unpack_dataset
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_checkpoint_path = None  # Initialize path to None
        self.is_initialized = False # Add a flag to track initialization for logger availability
        
        # Set number of epochs
        self.num_epochs = 1


    def set_initial_checkpoint(self, path: str):
        """Sets the path for the initial checkpoint to be loaded."""
        self.initial_checkpoint_path = path
        # Logger might not be initialized yet, print to stdout as a fallback
        log_message = f"Initial checkpoint path set to: {self.initial_checkpoint_path}"
        if hasattr(self, 'logger') and self.logger is not None:
            self.print_to_log_file(log_message)
        else:
            print(log_message)


    def on_train_start(self):
        # Call superclass's on_train_start() to initialize dataloaders, logger, etc.
        # This also initializes self.network if not already done.
        super().on_train_start()
        self.is_initialized = True # Mark as initialized for logging purposes
        self.print_to_log_file(f"CustomTrainer: Setting number of epochs to {self.num_epochs}")


        # Load custom network weights if an initial_checkpoint_path is provided
        if self.initial_checkpoint_path:
            self.print_to_log_file(f"Attempting to load initial network weights from: {self.initial_checkpoint_path} using load_pretrained_weights utility.")
            try:
                # Ensure the network is initialized before trying to load weights into it
                if self.network is None:
                    self.print_to_log_file("ERROR: self.network is None. Ensure trainer.initialize() has been called and network is built before on_train_start.")
                    # Or, you might need to call parts of initialize() here if it's guaranteed not to have run.
                    # However, super().on_train_start() should have called self.initialize() if was_initialized is False.
                else:
                    load_pretrained_weights(self.network, self.initial_checkpoint_path, verbose=True)
                    self.print_to_log_file(f"Successfully loaded initial network weights from {self.initial_checkpoint_path} using load_pretrained_weights.")
                    self.print_to_log_file("Optimizer state and grad_scaler state from this initial checkpoint are intentionally not loaded by this method.")
            
            except FileNotFoundError:
                self.print_to_log_file(f"ERROR: Initial checkpoint file not found at {self.initial_checkpoint_path}. Proceeding without loading these weights.")
            except Exception as e:
                self.print_to_log_file(f"ERROR loading initial network weights from {self.initial_checkpoint_path} using load_pretrained_weights: {e}. Proceeding without loading these weights.")
        else:
            self.print_to_log_file("No initial checkpoint path provided via `set_initial_checkpoint`; network will use weights from `initialize()` or train from scratch.")

        # Run a validation epoch before training starts
        self.print_to_log_file("Running initial validation epoch before actual training begins...")
        
        if self.dataloader_val is None:
            self.print_to_log_file("Validation dataloader (self.dataloader_val) is not initialized. Skipping initial validation.")
        else:
            self.print_to_log_file("Validation dataloader is initialized, calling on_validation_epoch_start() to prepare for validation.")
            self.on_validation_epoch_start()  # Sets network to eval mode and other necessary preparations
            self.print_to_log_file("on_validation_epoch_start() completed, proceeding with initial validation.")
            val_outputs = []
            # Store original training mode and set to eval
            self.print_to_log_file("Setting network to eval mode for initial validation.")
            original_training_state = self.network.training
            self.network.eval()
            self.print_to_log_file("Network set to eval mode, starting initial validation...")

            with torch.no_grad():  # Disable gradient calculations during validation
                for batch_id in range(self.num_val_iterations_per_epoch):
                    try:
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                        self.print_to_log_file(f"Initial validation: processing batch {batch_id + 1}/") # Optional: for verbose logging
                        
 
                    except Exception as e:
                        self.print_to_log_file(f"Error during initial validation_step for batch {batch_id}: {e}")
            
            if val_outputs:
                self.on_validation_epoch_end(val_outputs)
                self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
                self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
            else:
                self.print_to_log_file("No batches were processed or no outputs generated during initial validation.")
            
            # Restore original network mode
            self.network.train(original_training_state)
        
        self.print_to_log_file("Initial validation epoch finished.")

        # Ensure the network is in training mode for the upcoming training epochs
        self.network.train()
        self.print_to_log_file("Network set to train mode, ready for training.")

    def get_training_transforms(self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                              do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
                              regions, ignore_label):
        # ...existing code...
        default_transforms_obj = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label
        )
        
        tr_transforms_list = default_transforms_obj.transforms

        # Corrected: Insert AddPromptChannelsTransform before DownsampleSegForDSTransform
        # or before 'seg' is renamed to 'target' if deep supervision is off.
        # This ensures prompts are added to augmented 'data' using augmented 'seg'.
        
        # Find the index to insert the custom transform.
        # We want it after spatial/intensity augmentations but before 'seg' is processed for deep supervision targets.
        insertion_idx = len(tr_transforms_list) # Default to end if specific transforms not found


        for i, transform_instance in enumerate(tr_transforms_list):
            if isinstance(transform_instance, DownsampleSegForDSTransform):
                insertion_idx = i
                break

        tr_transforms_list.insert(insertion_idx, AddPromptChannelsTransform())
        
        return ComposeTransforms(tr_transforms_list)


    def get_validation_transforms(self, deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label):
        default_transforms_obj = super().get_validation_transforms(
            deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label
        )
        
        val_transforms_list = default_transforms_obj.transforms

        insertion_idx = len(val_transforms_list)
        for i, transform_instance in enumerate(val_transforms_list):
            if isinstance(transform_instance, DownsampleSegForDSTransform):
                insertion_idx = i
                break


        val_transforms_list.insert(insertion_idx, AddPromptChannelsTransform())
        return ComposeTransforms(val_transforms_list)
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 7,
            2,  # nnunet handles one class segmentation still as CE so we need 2 outputs.
            enable_deep_supervision
        )