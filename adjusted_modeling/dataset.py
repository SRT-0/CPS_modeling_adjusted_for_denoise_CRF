"""Definition of the datasets and associated functions."""
import os
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

__all__ = [
  "get_slice_points",
  "reconstruct_patched",
  "collate_fn",
  "MicrographDataset",
  "MicrographDatasetSingle",
  "MicrographDatasetEvery"
]

def create_weight_map(crop_size, bandwidth=None):
    """Create a Gaussian-like weight map which emphasizes the center of the patch.
    
    Args:
        crop_size (int): The size of the crop (assumed to be square for simplicity).
        bandwidth (int, optional): Defines the transition area from edge to center.
                                   Smaller values make the transition sharper.
                                   If None, it defaults to crop_size // 4.
    """
    if bandwidth is None:
        bandwidth = crop_size // 16  # Default bandwidth

    ramp = torch.linspace(0, 1, bandwidth)
    ramp = torch.cat([ramp, torch.ones(crop_size - 2 * bandwidth), ramp.flip(0)])
    weight_map = ramp[:, None] * ramp[None, :]
    return weight_map


def get_slice_points(image_size, crop_size, overlap):
    """ Calculate the slice points for cropping the image into overlapping patches. """
    step = crop_size - overlap
    num_points = (image_size - crop_size) // step + 1
    last_point = image_size - crop_size
    points = torch.arange(0, step * num_points, step)
    if points[-1] != last_point:
        points = torch.cat([points, torch.tensor([last_point])])
    return points


def reconstruct_patched(images, structured_grid, bandwidth=None):
    if bandwidth is None:
        bandwidth = images.shape[2] // 16  # Adjust this based on your needs

    weight_map = create_weight_map(images.shape[2], bandwidth).to(images.device)
    max_height = structured_grid[0, -1] + images.shape[2]
    max_width = structured_grid[1, -1] + images.shape[3]
    reconstructed_image = torch.zeros((images.shape[1], max_height, max_width), device=images.device)
    weights = torch.zeros_like(reconstructed_image)

    # Process in batches
    batch_size = 32  # Adjust this depending on your GPU capacity
    num_batches = (images.shape[0] + batch_size - 1) // batch_size  # Compute number of batches

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, images.shape[0])
        batch_images = images[batch_start:batch_end]
        batch_structured_grid = structured_grid[:, batch_start:batch_end]

        for idx, (start_i, start_j) in enumerate(zip(batch_structured_grid[0].flatten(), batch_structured_grid[1].flatten())):
            end_i = start_i + images.shape[2]
            end_j = start_j + images.shape[3]
            reconstructed_image[:, start_i:end_i, start_j:end_j] += batch_images[idx] * weight_map
            weights[:, start_i:end_i, start_j:end_j] += weight_map

    reconstructed_image /= weights.clamp(min=1)
    return reconstructed_image

### adjusted for pairwise denoised micrograph ↓
def collate_fn(batch):
    """
    Robust collate function that handles:
    1. Training patches (List of tuples) -> Flattens them
    2. Validation/Grid images (Tuples of 5 items) -> Stacks them
    3. Missing Denoised images (None) -> Returns None safely
    """
    
    # Case 1: Training with Patches (MicrographDataset)
    if isinstance(batch[0], list):
        images, denoiseds, masks = [], [], []
        for b in batch:
            for item in b:
                img, dnzd, mask = item
                images.append(img)
                denoiseds.append(dnzd)
                masks.append(mask)

        images = torch.stack(images)
        masks = torch.stack(masks)

        if all(d is None for d in denoiseds):
            denoiseds = None
        else:
            denoiseds = torch.stack(denoiseds)

        return images, denoiseds, masks
    
    # Case 2: Validation/Prediction (MicrographDatasetEvery)
    elif not isinstance(batch, list):
        return batch

    # Case 3: Single Image Training (MicrographDatasetSingle)
    else:
        transposed = list(zip(*batch))
        images = torch.stack(transposed[0])
        masks = torch.stack(transposed[2])
        
        if all(d is None for d in transposed[1]):
            denoiseds = None
        else:
            denoiseds = torch.stack(transposed[1])
            
        return images, denoiseds, masks

class MicrographDataset(Dataset):
  """
  Dataset for cryo-EM dataset that returns multiple patches per image.
  """
  def __init__(self, image_dir, label_dir, denoised_dir=None, filenames=None, crop_size=(512, 512), num_patches=1, img_ext='.npy', crop=None):
      self.image_dir = image_dir
      self.label_dir = label_dir
      self.num_patches = num_patches
      self.crop_size = crop_size
      if filenames is not None:
          self.filenames = filenames
      else:
          self.filenames = sorted(os.listdir(image_dir))
      basenames = [os.path.splitext(filename)[0] for filename in self.filenames]
      self.images = [os.path.join(image_dir, basename + img_ext) for basename in basenames]
      self.labels = [os.path.join(label_dir, basename + '.png') for basename in basenames]

      if denoised_dir is not None:
          self.denoised_images = [os.path.join(denoised_dir, basename + img_ext) for basename in basenames]
      else:
          self.denoised_images = [None] * len(self.images)

      if crop is None:
          self.crop = transforms.CenterCrop(3840)  # Adjust based on your specific needs
      else:
          self.crop = crop

  def __len__(self):
      return len(self.images)
  
  ### adjusted for pairwise denoised micrograph ↓
  def __getitem__(self, idx):
      mask = TF.to_tensor(Image.open(self.labels[idx]).convert("L"))
      image = torch.from_numpy(np.load(self.images[idx]).reshape((-1, mask.shape[1], mask.shape[2])))  # Assume images are 4096x4096
      
      denoised = None
      if self.denoised_images[idx] is not None:
          denoised = torch.from_numpy(np.load(self.denoised_images[idx]).reshape((-1, mask.shape[1], mask.shape[2])))

      patches = []
      for _ in range(self.num_patches):
          image_cropped, mask_cropped, denoised_cropped = self.transform(image, mask, denoised)
          patches.append((image_cropped, denoised_cropped, mask_cropped.long()))
      return patches

  ### adjusted for pairwise denoised micrograph ↓
  def transform(self, image, mask, denoised=None):
      if self.crop:
          image = self.crop(image)
          mask = self.crop(mask)
      
      i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
      image = TF.crop(image, i, j, h, w)
      mask = TF.crop(mask, i, j, h, w)
      if denoised is not None:
          denoised = TF.crop(denoised, i, j, h, w)

      #mask = torch.concat([1 - mask, mask], dim=0)  # For two-class problem: background and foreground
      
      return image, mask, denoised


class MicrographDatasetSingle(Dataset):
    """
    Dataset for cryo-EM with a single random crop per image.
    If label_dir=None, returns a zero mask instead.
    """
    def __init__(self, image_dir, label_dir=None, denoised_dir=None,  filenames=None,
                 crop_size=(512, 512), img_ext='.npy', crop=3840):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.denoised_dir = denoised_dir

        self.filenames = filenames if filenames is not None else sorted(os.listdir(image_dir))
        basenames = [os.path.splitext(f)[0] for f in self.filenames]
        self.images = [os.path.join(image_dir, b + img_ext) for b in basenames]
        if label_dir is not None:
            self.labels = [os.path.join(label_dir, b + '.png') for b in basenames]
        else:
            self.labels = [None] * len(basenames)

        if denoised_dir is not None:
            self.denoised_images = [os.path.join(denoised_dir, b + img_ext) for b in basenames]
        else:
            self.denoised_images = [None] * len(basenames)

        self.crop = transforms.CenterCrop(3840) if crop else None
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.label_dir is not None:
            mask = TF.to_tensor(Image.open(self.labels[idx]).convert("L"))
        else:
            img_np = np.load(self.images[idx])
            H, W = img_np.shape[-2], img_np.shape[-1]
            mask = torch.zeros((1, H, W), dtype=torch.uint8)

        if self.denoised_images[idx] is not None:
            den_np = np.load(self.denoised_images[idx])
            denoised = torch.from_numpy(den_np.reshape((-1, mask.shape[1], mask.shape[2])))
        else : 
            denoised = None

        img_np = np.load(self.images[idx])
        image = torch.from_numpy(img_np.reshape((-1, mask.shape[1], mask.shape[2])))
        return self.transform(image, mask, denoised)
    
    ### adjusted for pairwise denoised micrograph ↓
    def transform(self, image, mask, denoised=None):
        if self.crop:
            image = self.crop(image)
            mask = self.crop(mask)
            if denoised is not None:
                denoised = self.crop(denoised)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        if denoised is not None:
            denoised = TF.crop(denoised, i, j, h, w)
        return image, denoised, mask.long()
        
class MicrographDatasetEvery(MicrographDatasetSingle):
  """
  Dataset for cryo-EM dataset.
  The micrographs and ground truths will be divided into grid.
  """
  def __init__(self, *arg, **kwarg):
    super().__init__(*arg, **kwarg)
 
  ### adjusted for pairwise denoised micrograph ↓
  def transform(self, image, mask, denoised=None):
    if self.crop:
        image = self.crop(image)
        mask = self.crop(mask) #CenterCrop
        if denoised is not None:
            denoised = self.crop(denoised)

    image_dims = (image.size(-2), image.size(-1))
    crop_dims = (self.crop_size[-2], self.crop_size[-1])
    overlap_size = 64
    # Cache grid calculations to avoid redundancy
    #if (image_dims, crop_dims) not in self.slice_points_cache:
    grid_i = get_slice_points(image.size(-2), self.crop_size[-2], overlap_size)
    grid_j = get_slice_points(image.size(-1), self.crop_size[-1], overlap_size)
    grid = torch.cartesian_prod(grid_i, grid_j)
    #self.slice_points_cache[(image_dims, crop_dims)] = grid

    # Pre-allocate tensors for images and masks
    num_patches = grid.size(0)
    images = torch.zeros((num_patches, 1, *self.crop_size), device=image.device, dtype=image.dtype)
    masks = torch.zeros((num_patches, 1, *self.crop_size), device=mask.device, dtype=mask.dtype)

    if denoised is not None:
        denoiseds = torch.zeros((num_patches, denoised.shape[0], *self.crop_size),
                                       device=denoised.device, dtype=denoised.dtype)
    else:
        denoiseds = None

    for idx, (i, j) in enumerate(grid):
        ii = i.item(); jj = j.item()
        images[idx] = TF.crop(image, ii, jj, self.crop_size[-2], self.crop_size[-1])
        masks[idx] = TF.crop(mask, ii, jj, self.crop_size[-2], self.crop_size[-1])
        if denoised is not None:
            denoiseds[idx] = TF.crop(denoised, ii, jj, self.crop_size[-2], self.crop_size[-1])

    structured_grid = torch.stack([grid[:, 0], grid[:, 1]], dim=0)

    #masks = torch.concat([1-masks, masks], dim=1) # Remove this line if background is not consider.

    return images, denoiseds, masks.long(), structured_grid, mask.long()