from typing import Union, List, Tuple
from flask import g
import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from typing import Union, List, Tuple
import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform

class NormalizeSingleImageTransform(AbstractTransform):
    def __call__(self, **data_dict):
        imgs = data_dict["image"]
        assert imgs.shape[0] == 1
        means = torch.mean(imgs, dim=list(range(1, imgs.ndim)), keepdim=True)
        stds = torch.std(imgs, dim=list(range(1, imgs.ndim)), keepdim=True)
        eps = 1e-6
        normed_imgs = (imgs - means) / (stds + eps)
        data_dict["image"] = normed_imgs
        segs = data_dict["segmentation"]
        if torch.min(segs) < 0:
            #print("Segmentation contains negative values, adjusting them")
            #print(f"Min: {torch.min(segs)}, Max: {torch.max(segs)}")
            data_dict["segmentation"] = data_dict["segmentation"] + 1 
        return data_dict


def mask2D_to_bbox(
    gt2D,
):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, 6, 1)[0]
    scale_y, scale_x = gt2D.shape
    bbox_shift_x = int(bbox_shift * scale_x / 256)
    bbox_shift_y = int(bbox_shift * scale_y / 256)
    # print(f'{bbox_shift_x=} {bbox_shift_y=} with orig {bbox_shift=}')
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W - 1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H - 1, y_max + bbox_shift_y)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes


def mask3D_to_bbox(
    gt3D,
):
    b_dict = {}
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    z_indices = np.unique(z_indices)
    # middle of z_indices
    z_middle = z_indices[len(z_indices) // 2]

    D, H, W = gt3D.shape
    b_dict["z_min"] = z_min
    b_dict["z_max"] = z_max
    b_dict["z_mid"] = z_middle

    gt_mid = gt3D[z_middle]

    box_2d = mask2D_to_bbox(
        gt_mid,
    )
    x_min, y_min, x_max, y_max = box_2d
    b_dict["z_mid_x_min"] = x_min
    b_dict["z_mid_y_min"] = y_min
    b_dict["z_mid_x_max"] = x_max
    b_dict["z_mid_y_max"] = y_max

    assert z_min == max(0, z_min)
    assert z_max == min(D - 1, z_max)
    return b_dict


class AddBBoxAndEmptyChannelsTransform(AbstractTransform):
    def __init__(self, class_idx=None):
        """
        Takes the input data and adds 7 prompt channels to it, fills bbox into channel 1/-6:
        - Channel 0: Previous segmentation
        - Channel -6: Positive bounding box/lasso
        - Channel -5: Negative bounding box/lasso
        - Channel -4: Positive point interaction
        - Channel -3: Negative point interaction
        - Channel -2: Positive scribble
        - Channel -1: Negative scribble

        """
        print("Initializing AddBBoxAndEmptyChannelsTransform again v3")
        self.class_idx = class_idx
        self.verbose = False  # Set to False to disable debug prints

    def __call__(self, **data_dict):
        # now the way should be:
        # add bbox always (is always there)
        # maybe sometimes 2d, sometimes 3d? or just try first 2d and then see 3d?
        # compare to evaluation with tidianes nn predictor, not yours

        ## bnow next is happening inside model:
        # then decide randomly on the number of extra steps, uniform from 1 to 5
        # clicks radius take from competition evaluation script
        # correspondingly add negative or positive clicks
        # add prior segmentation always as initial segmentation guess

        # Get the input image and ground truth
        imgs = data_dict["image"]
        gts = data_dict["segmentation"]

        # Debug: print shapes and unique values
        if self.verbose:
            print(f"imgs shape: {imgs.shape}, gts shape: {gts.shape}")
            print(f"gts unique values: {np.unique(gts)}")
        
        # pick a random class in the gts

        unique_classes = torch.unique(gts).cpu().numpy()
        gts_np = gts.cpu().numpy().squeeze()  # Convert to numpy and remove batch dimension if present
        

        unique_classes = unique_classes[unique_classes > 0]  # Remove background class (0)

        if len(unique_classes) == 0:
            print("No foreground classes found, using background")
            class_idx = 0
        else:
            class_idx = np.random.choice(unique_classes)
        # for debugging
        if self.class_idx is not None:
            class_idx = self.class_idx

        gt_class = (gts_np == class_idx).astype(np.float32)

        box_dict = mask3D_to_bbox(
            gt_class,
        )
        prompt_channels = torch.zeros((7, *imgs.shape[1:]), device=imgs.device)
        prompt_channels[
            1,
            box_dict["z_min"] : box_dict["z_max"] + 1,
            box_dict["z_mid_y_min"] : box_dict["z_mid_y_max"] + 1,
            box_dict["z_mid_x_min"] : box_dict["z_mid_x_max"] + 1,
        ] = 1
        """
        # just for now to check how good it is
        prompt_channels[
            1,
            (box_dict["z_min"] + box_dict["z_max"]) // 2,
            box_dict["z_mid_y_min"] : box_dict["z_mid_y_max"] + 1,
            box_dict["z_mid_x_min"] : box_dict["z_mid_x_max"] + 1,
        ] = 1
        """
        # Convert to torch tensor
        gt_class_torch = torch.from_numpy(gt_class).to(imgs.device)

        # Concatenate the original image with the prompt channels,
        # also add the ground turth needed by the model wrapper to generate clicks
        # will not be seen by the model itself at all, will be removed by model wrapper before
        data_dict["image"] = torch.cat([imgs, prompt_channels], dim=0)
        data_dict["segmentation"] = gt_class_torch.unsqueeze(0)  # Add batch dimension
        return data_dict


class AddBBoxAndEmptyChannelsSingleClassTransform(AbstractTransform):
    def __init__(self, class_idx=None):
        """
        Takes the input data and adds 7 prompt channels to it, fills bbox into channel 1/-6:
        - Channel 0: Previous segmentation
        - Channel -6: Positive bounding box/lasso
        - Channel -5: Negative bounding box/lasso
        - Channel -4: Positive point interaction
        - Channel -3: Negative point interaction
        - Channel -2: Positive scribble
        - Channel -1: Negative scribble

        """
        print("Initializing AddBBoxAndEmptyChannelsSingleClassTransform")
        self.class_idx = class_idx

    def __call__(self, **data_dict):
        # now the way should be:
        # add bbox always (is always there)
        # maybe sometimes 2d, sometimes 3d? or just try first 2d and then see 3d?
        # compare to evaluation with tidianes nn predictor, not yours

        ## bnow next is happening inside model:
        # then decide randomly on the number of extra steps, uniform from 1 to 5
        # clicks radius take from competition evaluation script
        # correspondingly add negative or positive clicks
        # add prior segmentation always as initial segmentation guess

        # Get the input image and ground truth
        imgs = data_dict["image"]
        gts = data_dict["segmentation"]

        this_coords = torch.argwhere(gts > 0)

        i_starts, i_stops = (
            torch.min(this_coords, axis=0).values,
            torch.max(this_coords, axis=0).values + 1,
        )

        prompt_channels = torch.zeros((7, *imgs.shape[1:]), device=imgs.device)
        # prompt_channels[
        #     1,
        #     box_dict["z_min"] : box_dict["z_max"] + 1,
        #     box_dict["z_mid_y_min"] : box_dict["z_mid_y_max"] + 1,
        #     box_dict["z_mid_x_min"] : box_dict["z_mid_x_max"] + 1,
        # ] = 1

        # just for now to check how good it is
        prompt_channels[
            1,
            (i_starts[0] + i_stops[0]) // 2,
            i_starts[1] : i_stops[1],
            i_starts[2] : i_stops[2],
        ] = 1

        # Concatenate the original image with the prompt channels,
        # also add the ground turth needed by the model wrapper to generate clicks
        # will not be seen by the model itself at all, will be removed by model wrapper before
        data_dict["image"] = torch.cat([imgs, prompt_channels], dim=0)
        data_dict["segmentation"] = gts
        return data_dict


class AddSegToImageTransform(AbstractTransform):
    def __init__(
        self,
    ):
        """
        Add segmentation to image as last channel.

        """
        print("Initializing AddSegToImageTransform")

    def __call__(self, **data_dict):
        imgs = data_dict["image"]
        # due to deep supervision there are multiple ground truth masks, first one is full one
        gts = data_dict["segmentation"]
        data_dict["image"] = torch.cat([imgs, gts], dim=0)
        return data_dict
