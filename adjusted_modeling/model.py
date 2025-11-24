import torch
import torch.nn as nn
from collections import OrderedDict
from convcrf import GaussCRF, ConvCRF


def create_model(backbone, addout=True):
    """
    Creates a model wrapper for the backbone network.
    
    Args:
        backbone: The backbone network (e.g., UNet, DeepLabV3)
        addout (bool): Whether to add output wrapper. Defaults to True.
        
    Returns:
        Model with or without output wrapper
    """
    if addout:
        model = Model_Out(backbone)
    else:
        model = backbone
    return model


def create_crf_model(backbone, config, shape, num_classes, use_gpu=False, freeze_backbone=False):
    """
    Creates a model combining backbone network with Gaussian CRF layer.
    
    Args:
        backbone: The backbone segmentation network
        config: CRF configuration parameters
        shape: Input image shape (height, width)
        num_classes (int): Number of segmentation classes
        use_gpu (bool): Whether to use GPU acceleration. Defaults to False.
        freeze_backbone (bool): Whether to freeze backbone parameters. Defaults to False.
        
    Returns:
        ModelWithGausscrf: Combined model with CRF post-processing
    """
    if freeze_backbone:
        for params in backbone.parameters():
          params.requires_grad = False
    model = ModelWithGausscrf(backbone, config=config, shape=shape, num_classes=num_classes, use_gpu=use_gpu)
    return model


class Model_Out(nn.Module):
    """
    Wrapper class that standardizes backbone model output format.
    Ensures consistent output structure across different backbone architectures.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        """
        Forward pass through the backbone network.
        
        Args:
            x (torch.Tensor): Batch of input images [B, C, H, W]
            
        Returns:
            OrderedDict: Dictionary containing 'out' key with logits
        """
        logits = self.backbone(x)
        return OrderedDict([
        ('out', logits)
      ])

class ModelWithGausscrf(nn.Module):
    """
    Combined model that applies Gaussian CRF post-processing to backbone predictions.
    The CRF layer refines segmentation boundaries using spatial consistency constraints.
    """
    def __init__(self, backbone, config, shape, num_classes, use_gpu=False):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.num_classes = num_classes
        self.shape = shape
        self.use_gpu = use_gpu
        self.gausscrf = GaussCRF(conf=self.config, shape=self.shape,
                                 nclasses=self.num_classes, use_gpu= self.use_gpu)
        
    ### adjusted for pairwise denoised micrograph ↓
    def forward(self, x, pairwise_img=None):
        """
        Forward pass with CRF post-processing.
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            
        Returns:
            OrderedDict: Contains 'backbone' (raw predictions) and 'out' (CRF-refined)
        """
        unary = self.backbone(x)['out']
        return OrderedDict([
          ('backbone', unary),
          ('out', self.gausscrf(unary, x, pairwise_img))
        ])


try:
    import CRF
    
    ### adjusted for pairwise denoised micrograph ↓
    class ModelWithFWCRF(nn.Module):
        """Combined Model class for UNET with configurable Frank-Wolfe CRF."""
        def __init__(self, backbone, crf, use_unary_only=False):
            super().__init__()
            self.backbone = backbone
            self.crf = crf
            self.use_unary_only = use_unary_only

        def forward(self, x, pairwise_img=None):
            """Forward pass for input batch of images."""
            unary = self.backbone(x)
            image_for_crf = pairwise_img if pairwise_img is not None else x
            if self.use_unary_only:
                return {'backbone': unary, 'out': self.crf(unary, unary)}
            else:
                return {'backbone': unary, 'out': self.crf(image_for_crf, unary)}

    def create_fwcrf_model(backbone, crf, use_unary_only=False):
        """Factory function to create a UNET model with Frank-Wolfe CRF."""
        return ModelWithFWCRF(backbone, crf, use_unary_only)

    def setup_crf(solver, num_classes):
        """Setup CRF based on the solver type."""
        if solver not in ['fw', 'mf']:
            raise NotImplementedError("Solver not supported")
        
        crf = CRF.DenseGaussianCRF(
            classes=num_classes,
            alpha=160,
            beta=0.05,
            gamma=3.0,
            spatial_weight=1.0,
            bilateral_weight=1.0,
            compatibility=1.0,
            init='potts',
            solver=solver,
            iterations=5,
            params=None if solver == 'mf' else CRF.FrankWolfeParams(
                scheme='fixed', stepsize=1.0, regularizer='l2', lambda_=1.0,
                lambda_learnable=False, x0_weight=0.5, x0_weight_learnable=False)
        )
        return crf
    
  # class ModelWithFWCRF(nn.Module):
      # def __init__(self, backbone, crf):
          # super().__init__()
          # self.backbone = backbone
          # self.crf = crf

      # def forward(self, x):
          # """
          # x is a batch of input images
          # """
          # unary = self.backbone(x)['out']
          # logits = self.crf(x, unary)
          # return OrderedDict([
          # ('backbone', unary),
          # ('out', logits)
        # ])
        
  # class ModelWithFWCRF_UNET(nn.Module):
      # def __init__(self, backbone, crf):
          # super().__init__()
          # self.backbone = backbone
          # self.crf = crf

      # def forward(self, x):
          # """
          # x is a batch of input images
          # """
          # unary = self.backbone(x)
          # logits = self.crf(x, unary)
          # return OrderedDict([
          # ('backbone', unary),
          # ('out', logits)
        # ])

  # def create_fwcrf_model_unet(backbone, params, num_classes, alpha=160, beta=0.05, gamma=3.0, iterations=5, freeze_backbone=False):
    # if freeze_backbone:
      # for param in backbone.parameters():
        # param.requires_grad = False
    # crf = CRF.DenseGaussianCRF(
            # classes=num_classes,
            # alpha=alpha,
            # beta=beta,
            # gamma=gamma,
            # spatial_weight=1.0,
            # bilateral_weight=1.0,
            # compatibility=1.0,
            # init='potts',
            # solver='mf',
            # iterations=iterations,
            # x0_weight = 0,
            # params=params)
    # model = ModelWithFWCRF_UNET(backbone, crf)
    # return model

  # def create_fwcrf_model(backbone, params, num_classes, alpha=160, beta=0.05, gamma=3.0, iterations=5, freeze_backbone=False):
    # if freeze_backbone:
      # for param in backbone.parameters():
        # param.requires_grad = False
    # crf = CRF.DenseGaussianCRF(
            # classes=num_classes,
            # alpha=alpha,
            # beta=beta,
            # gamma=gamma,
            # spatial_weight=1.0,
            # bilateral_weight=1.0,
            # compatibility=1.0,
            # init='potts',
            # solver='mf',
            # iterations=iterations,
            # x0_weight = 0,
            # params=params)
    # model = ModelWithFWCRF(backbone, crf)
    # return model
except:
    pass