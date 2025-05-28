import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.prompt_channels_transform import AddPromptChannelsTransform
from typing import Union, Tuple, List
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
import numpy as np
from torch import nn
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights # Import the function
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.data_augmentation.custom_transforms.custom_transforms import (
    AddBBoxAndEmptyChannelsTransform,
    AddSegToImageTransform,
    NormalizeSingleImageTransform,
)
from nnunetv2.training.nnUNetTrainer.model_wrap import ModelPrevSegAndClickWrapper

class CustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Ensure all original args for nnUNetTrainer are passed, especially unpack_dataset
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_checkpoint_path = None  # Initialize path to None
        self.is_initialized = False # Add a flag to track initialization for logger availability
        
        # Set number of epochs
        self.num_epochs = 10
        self.initial_lr = 1e-3
        self.freeze_decoder = False 
    
    def configure_optimizers(self):
        # Example: Freeze decoder layers
        if hasattr(self.network, 'decoder') and self.freeze_decoder:
            self.print_to_log_file("Freezing decoder layers for fine-tuning.")
            for param in self.network.decoder.parameters():
                param.requires_grad = False
            
            # Ensure the network module is used if DDP is active
            network_to_get_params_from = self.network.module if self.is_ddp else self.network
            
            # Filter parameters that require gradients (i.e., encoder and any other unfrozen layers)
            trainable_params = filter(lambda p: p.requires_grad, network_to_get_params_from.parameters())
            
            # Count trainable parameters for logging
            num_trainable_params = sum(p.numel() for p in trainable_params if p.requires_grad)
            # Re-filter for the optimizer after counting, as the filter object gets consumed
            trainable_params_for_optimizer = filter(lambda p: p.requires_grad, network_to_get_params_from.parameters())

            self.print_to_log_file(f"Number of trainable parameters after freezing decoder: {num_trainable_params}")

            optimizer = torch.optim.AdamW(trainable_params_for_optimizer, lr=self.initial_lr, weight_decay=self.weight_decay)
        else:
            self.print_to_log_file("Decoder attribute not found, training all layers.")
            # Default behavior if no specific layer freezing logic or 'decoder' attribute isn't present
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def set_initial_checkpoint(self, path: str, freeze_decoder: bool = True):
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
        # This also initializes self.network by calling self.initialize(), which in turn
        # calls self.build_network_architecture().
        super().on_train_start()
        self.is_initialized = True # Mark as initialized for logging purposes
        self.print_to_log_file(f"CustomTrainer: Setting number of epochs to {self.num_epochs}")


        # Load custom network weights if an initial_checkpoint_path is provided
        if self.initial_checkpoint_path:
            self.print_to_log_file(f"Attempting to load initial network weights from: {self.initial_checkpoint_path} into the underlying original network.")
            try:
                if self.network is None:
                    self.print_to_log_file("ERROR: self.network is None. Ensure trainer.initialize() has been called and network is built before on_train_start.")
                else:
                    # Determine the target network (the bare model inside the wrapper)
                    target_load_network = self.network.module.orig_network if self.is_ddp else self.network.orig_network
                    
                    load_pretrained_weights(target_load_network, self.initial_checkpoint_path, verbose=True)
                    self.print_to_log_file(f"Successfully loaded initial network weights into the underlying original network from {self.initial_checkpoint_path}.")
                    self.print_to_log_file("Optimizer state and grad_scaler state from this initial checkpoint are intentionally not loaded by this method if using load_pretrained_weights directly.")
            
            except FileNotFoundError:
                self.print_to_log_file(f"ERROR: Initial checkpoint file not found at {self.initial_checkpoint_path}. Proceeding without loading these weights.")
            except Exception as e:
                self.print_to_log_file(f"ERROR loading initial network weights into the underlying original network from {self.initial_checkpoint_path}: {e}. Proceeding without loading these weights.")
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
                        self.print_to_log_file(f"Initial validation: processing batch {batch_id + 1}/{self.num_val_iterations_per_epoch}") # Optional: for verbose logging
                        
 
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
        default_transforms_obj = super().get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label
        )
        
        tr_transforms_list = default_transforms_obj.transforms
        
        #Insert channels transforms at the beginning before any downsampling transforms.
        insertion_idx = len(tr_transforms_list) # Default to end if specific transforms not found


        for i, transform_instance in enumerate(tr_transforms_list):
            if isinstance(transform_instance, DownsampleSegForDSTransform):
                insertion_idx = i
                break

        
        
        tr_transforms_list.insert(0, NormalizeSingleImageTransform())
        #tr_transforms_list.insert(insertion_idx, AddPromptChannelsTransform())
        tr_transforms_list.insert(insertion_idx, AddBBoxAndEmptyChannelsTransform())
        tr_transforms_list.insert(insertion_idx + 1, AddSegToImageTransform())
        
        
        return ComposeTransforms(tr_transforms_list)


    def get_validation_transforms(self, deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label):
        default_transforms_obj = super().get_validation_transforms(
            deep_supervision_scales, is_cascaded, foreground_labels, regions, ignore_label
        )
        
        val_transforms_list = default_transforms_obj.transforms
        #Insert channels transforms at the beginning before any downsampling transforms.
        
        insertion_idx = len(val_transforms_list)
        for i, transform_instance in enumerate(val_transforms_list):
            if isinstance(transform_instance, DownsampleSegForDSTransform):
                insertion_idx = i
                break
        
        val_transforms_list.insert(0, NormalizeSingleImageTransform())
        #val_transforms_list.insert(insertion_idx, AddPromptChannelsTransform())
        val_transforms_list.insert(insertion_idx, AddBBoxAndEmptyChannelsTransform())
        val_transforms_list.insert(insertion_idx+1,AddSegToImageTransform())
        return ComposeTransforms(val_transforms_list)
    
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # First, build the bare network using the parent's method
        # We pass num_input_channels + 7 to the superclass method because the bare network
        # needs to be configured for these input channels.
        bare_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 7, # This is the total number of channels the bare_network will expect
            2,  # nnunet handles one class segmentation still as CE so we need 2 outputs.
            enable_deep_supervision,
        )

        print(f"Wrapping the network of type: {bare_network.__class__} with ModelPrevSegAndClickWrapper")
        # Then, wrap this bare_network with your custom wrapper
        return ModelPrevSegAndClickWrapper(bare_network)
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Saves the checkpoint.
        This method is overridden to save the state_dict of self.network.orig_network
        (or self.network.module.orig_network for DDP) directly, so that keys do not
        have the 'orig_network.' prefix.
        """
        if not self.is_initialized: # Use the flag you defined
            self.print_to_log_file("Cannot save checkpoint. Trainer not initialized yet. "
                                   "Ensure super().on_train_start() or trainer.initialize() has been called.",
                                   also_print_to_console=True, add_timestamp=False)
            return

        if self.local_rank == 0: # Checkpoint saving should only happen on rank 0
            self.print_to_log_file(f"Saving checkpoint to {filename}...")

            # Determine the network whose state_dict we want to save (the bare model)
            if self.is_ddp:
                if not hasattr(self.network.module, 'orig_network'):
                    self.print_to_log_file("ERROR: DDP self.network.module does not have 'orig_network' attribute. Cannot save checkpoint correctly.",
                                           also_print_to_console=True, add_timestamp=False)
                    return
                network_to_save = self.network.module.orig_network
            else:
                if not hasattr(self.network, 'orig_network'):
                    self.print_to_log_file("ERROR: self.network does not have 'orig_network' attribute. Cannot save checkpoint correctly.",
                                           also_print_to_console=True, add_timestamp=False)
                    return
                network_to_save = self.network.orig_network
            
            network_weights = network_to_save.state_dict()
            optimizer_weights = self.optimizer.state_dict()
            grad_scaler_weights = self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            
            # Reconstruct the saved_dict similar to nnUNetTrainer.save_checkpoint
            # but using the modified network_weights.
            # You might need to adjust what's included based on nnUNetTrainer's version.
            saved_dict = {
                'network_weights': network_weights,
                'optimizer_weights': optimizer_weights, # Or optimizer_state from base
                'grad_scaler_weights': grad_scaler_weights, # Or grad_scaler_state from base
                'current_epoch': self.current_epoch, # nnUNetTrainer saves current_epoch + 1
                '_best_ema_dice': self._best_ema_dice if hasattr(self, '_best_ema_dice') else self._best_ema, # adapt to your logger variable
                # Include other necessary items from nnUNetTrainer's save_checkpoint:
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                'logging': self.logger.get_checkpoint(),
                # Add these if you use them and they are in the base class save_checkpoint
                # 'best_validation_loss': self._best_validation_loss,
                # 'best_val_eval_criterion_MA': self._best_val_eval_criterion_MA,
                # 'best_val_eval_criterion_for_checkpoint': self._best_val_eval_criterion_for_checkpoint,
            }
            
            try:
                torch.save(saved_dict, filename)
                self.print_to_log_file(f"Checkpoint saved successfully to {filename} (weights from orig_network).")
            except Exception as e:
                self.print_to_log_file(f"Error saving checkpoint {filename}: {e}", also_print_to_console=True)
        else:
            # If not local_rank 0, still log that checkpointing is skipped for this rank
            self.print_to_log_file(f"Skipping checkpoint save on rank {self.local_rank}", also_print_to_console=False, add_timestamp=True)