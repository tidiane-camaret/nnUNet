from typing import Union, List, Tuple

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform


from typing import Union, List, Tuple

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class AddPromptChannelsTransform(AbstractTransform):
    def __init__(self, num_points: int = 5, point_radius: int = 2, generate_prev_segmentation: bool = False, verbose: bool = False):
        """
        Takes the input data and adds channels for interactive segmentation prompts. The channels are:
        - Channel 0: Previous segmentation
        - Channel -6: Positive bounding box/lasso
        - Channel -5: Negative bounding box/lasso
        - Channel -4: Positive point interaction
        - Channel -3: Negative point interaction
        - Channel -2: Positive scribble
        - Channel -1: Negative scribble
        
        Args:
            num_points: Number of points to sample (default: 5)
            point_radius: Radius of each point (default: 2)
            verbose: Whether to print debug information (default: False)
        """
        self.num_points = num_points
        self.point_radius = point_radius
        self.generate_prev_segmentation = generate_prev_segmentation
        self.verbose = verbose
        if self.verbose:
            print(f"Initializing AddPromptChannelsTransform")

    def __call__(self, **data_dict):
        if self.verbose:
            print(f"AddPromptChannelsTransform called with keys: {list(data_dict.keys())}")
        
        # Get the input image and ground truth
        imgs = data_dict['image']

        #print(len(data_dict['segmentation']))
        gts = data_dict['segmentation']
        
        # Debug: print shapes and unique values
        if self.verbose:
            print(f"imgs shape: {imgs.shape}, gts shape: {gts.shape}")
            print(f"gts unique values: {np.unique(gts)}")
        
        # pick a random class in the gts

        unique_classes = torch.unique(gts).cpu().numpy()
        gts_np = gts.cpu().numpy()

        unique_classes = unique_classes[unique_classes > 0]  # Remove background class (0)
        
        if len(unique_classes) == 0:
            if self.verbose:
                print("No foreground classes found, using background")
            class_idx = 0
        else:
            class_idx = np.random.choice(unique_classes)
        
        if self.verbose:
            print(f"Selected class: {class_idx}")
        gt_class = (gts_np == class_idx).astype(np.float32)
        
        # Remove batch dimension
        gt_class = gt_class.squeeze(0) 

        if self.verbose:
            print(f"gt_class final shape: {gt_class.shape}")
    
        
        # Convert to torch tensor
        gt_class_torch = torch.from_numpy(gt_class).to(imgs.device)
        
        # Create tensor for all prompt channels
        # Shape: (7, *imgs.shape[1:])
        prompt_channels = torch.zeros((7, *imgs.shape[1:]), device=imgs.device)

        #### BBOX PROMPT ####
        if self.verbose:
            print("Writing bounding box prompt only")
        
        # Channel 1 (-6): Positive bounding box/lasso
        if np.sum(gt_class) > 0:
            pos_indices = np.where(gt_class > 0)
            if self.verbose:
                print(f"gt_class shape: {gt_class.shape}, nb of positive voxels: {np.sum(gt_class > 0)}")
            z, y, x = pos_indices

            z_min, z_max = np.min(z), np.max(z)
            y_min, y_max = np.min(y), np.max(y)
            x_min, x_max = np.min(x), np.max(x)

            if self.verbose:
                print(f"Bounding box: z({z_min}, {z_max}), y({y_min}, {y_max}), x({x_min}, {x_max})")
            
            # Ensure indices are within bounds
            z_min = max(0, z_min)
            z_max = min(prompt_channels.shape[1]-1, z_max)
            y_min = max(0, y_min)
            y_max = min(prompt_channels.shape[2]-1, y_max)
            x_min = max(0, x_min)
            x_max = min(prompt_channels.shape[3]-1, x_max)
                
            # Fill channel 1 with 1s in the defined bounding box
            prompt_channels[1, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1

        #### PREV SEG AND CLICKS PROMPTS  ####
        
        if self.generate_prev_segmentation:
        
            # Simulate the previous segmentation by applying erosion
            # Convert to float and add batch and channel dimensions
            gt_float = gt_class_torch.float()
            
            # Ensure we have the right dimensions for max_pool3d (needs 5D: batch, channel, depth, height, width)

            gt_float = gt_float.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

            
            if self.verbose:
                print(f"gt_float shape before padding: {gt_float.shape}")
            
            # Check if tensor is empty
            if gt_float.numel() == 0:
                if self.verbose:
                    print("Error: gt_float is empty")
                prev_seg = torch.zeros_like(gt_class_torch)
            else:
                # Use max pooling for erosion
                # Pad the input (pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
                padded = torch.nn.functional.pad(gt_float, (1,1,1,1,1,1), mode='constant', value=0)
                if self.verbose:
                    print(f"Padded shape: {padded.shape}")
                
                # Apply max pooling with a 3x3x3 kernel
                # This will erode the object by removing pixels at the boundaries
                try:
                    prev_seg = torch.nn.functional.max_pool3d(
                        padded,
                        kernel_size=3,
                        stride=1,
                        padding=0
                    )
                    # Remove batch and channel dimensions
                    prev_seg = prev_seg.squeeze(0).squeeze(0)
                    if self.verbose:
                        print(f"prev_seg shape: {prev_seg.shape}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error in max_pool3d: {e}")
                        print(f"Padded tensor shape: {padded.shape}")
                        print(f"Padded tensor device: {padded.device}")
                        print(f"Padded tensor dtype: {padded.dtype}")
                    # Fallback: use original gt_class
                    prev_seg = gt_class_torch
            
            # Channel 0: Previous segmentation (initialize with prev_seg ground truth)
            prompt_channels[0] = prev_seg
            
            if np.sum(gt_class) > 0:  # Only if there are positive pixels
 

                # Generate point interactions
                # Get boundary points by subtracting prev_seg from original
                boundary = gt_class - prev_seg.cpu().numpy()
                boundary_indices = np.where(boundary > 0)
                
                if len(boundary_indices) == 3:  # 3D case
                    boundary_z, boundary_y, boundary_x = boundary_indices
                elif len(boundary_indices) == 4:  # 4D case - take last 3 dimensions
                    _, boundary_z, boundary_y, boundary_x = boundary_indices
                else:
                    boundary_z = boundary_y = boundary_x = np.array([])
                
                if len(boundary_z) > 0:
                    # Sample random points from boundary
                    indices = np.random.choice(len(boundary_z), min(self.num_points, len(boundary_z)), replace=False)
                    
                    # Add positive points (on boundary) - Channel 3 (-4)
                    for idx in indices:
                        z_pt, y_pt, x_pt = boundary_z[idx], boundary_y[idx], boundary_x[idx]
                        # Create a small sphere around the point
                        z_slice = slice(max(0, z_pt-self.point_radius), min(prompt_channels.shape[1], z_pt+self.point_radius+1))
                        y_slice = slice(max(0, y_pt-self.point_radius), min(prompt_channels.shape[2], y_pt+self.point_radius+1))
                        x_slice = slice(max(0, x_pt-self.point_radius), min(prompt_channels.shape[3], x_pt+self.point_radius+1))
                        prompt_channels[3, z_slice, y_slice, x_slice] = 1
                    
                    # Add negative points (outside the object) - Channel 4 (-3)
                    for _ in range(self.num_points):
                        # Sample a point outside the object but near the boundary
                        z_neg = np.random.randint(max(0, z_min-10), min(prompt_channels.shape[1], z_max+10))
                        y_neg = np.random.randint(max(0, y_min-10), min(prompt_channels.shape[2], y_max+10))
                        x_neg = np.random.randint(max(0, x_min-10), min(prompt_channels.shape[3], x_max+10))
                        
                        # Ensure point is outside the object
                        if (0 <= z_neg < gt_class.shape[-3] and 
                            0 <= y_neg < gt_class.shape[-2] and 
                            0 <= x_neg < gt_class.shape[-1]):
                            if gt_class[z_neg, y_neg, x_neg] == 0:  # only add if outside object
                                z_slice = slice(max(0, z_neg-self.point_radius), min(prompt_channels.shape[1], z_neg+self.point_radius+1))
                                y_slice = slice(max(0, y_neg-self.point_radius), min(prompt_channels.shape[2], y_neg+self.point_radius+1))
                                x_slice = slice(max(0, x_neg-self.point_radius), min(prompt_channels.shape[3], x_neg+self.point_radius+1))
                                prompt_channels[4, z_slice, y_slice, x_slice] = 1
        
        # Concatenate the original image with the prompt channels
        data_dict['image'] = torch.cat([imgs, prompt_channels], dim=0)
        data_dict['segmentation'] = gt_class_torch.unsqueeze(0)  # Add batch dimension

        
        
        if self.verbose:
            print(f"Output data shape: {data_dict['image'].shape}")
        return data_dict

class AddEmptyChannelsTransform(AbstractTransform):
    def __init__(self, num_empty_channels: int = 7, key: str = "data"):
        """
        Takes the input data and adds empty channels to it:

        Args:
            num_empty_channels: Number of channels to add (default: 7)
            key: The key in data_dict that contains the input data (default: "data")
        """
        self.num_empty_channels = num_empty_channels
        self.key = key
        print(f"Initializing AddEmptyChannelsTransform with {num_empty_channels} channels and key {key}")

    def __call__(self, **data_dict):
        print(f"AddEmptyChannelsTransform called with keys: {list(data_dict.keys())}")
        
        # Get the input image and ground truth
        imgs = data_dict['image']
        gts = data_dict['segmentation']
        
        # Create tensor for all channels
        # Shape: (7, *imgs.shape[1:])
        channels = torch.zeros((self.num_empty_channels, *imgs.shape[1:]), device=imgs.device)

        
        # Concatenate the original image with the interaction channels
        data_dict['image'] = torch.cat([imgs, channels], dim=0)
        
        print(f"Output data shape: {data_dict['image'].shape}")
        return data_dict