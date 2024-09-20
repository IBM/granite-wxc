import os
from typing import Optional

import torch
from torch import nn

from granitewxc.utils import distributed

class PatchEmbed(torch.nn.Module):
    '''
    Patch embedding via 2D convolution.
    same as weather-fm.models.hiera_maxvit.PatchEmbed
    '''

    def __init__(self, patch_size: int | tuple[int, ...], channels: int, embed_dim: int):
        super().__init__()

        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=1, bias=True, padding='same', padding_mode='replicate'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of shape [batch, channels, lat, lon].
        Returns:
            Tensor with shape [batch, embed_dim, lat//patch_size, lon//patch_size]
        '''

        x = self.proj(x)

        return x


class FinetuneWrapper(torch.nn.Module):
    """ General purpose wrapper class to finetune using us configurable head and backbone """

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module, config=None):
        super().__init__()

        self.backbone = backbone  # pre-trained model
        self.head = head  # task specific head
        self.config = config
        
        # weight_path = Path(config.load_model)
        # if os.path.isfile(weight_path):
        #     cp_path = weight_path
        # else:
        #     cp_path = weight_path / "train"
        # self.checkpointer = Checkpointer(
        #     cp_path, 2, "fsdp", 0, 0
        # )

    def forward(self, batch):
        """
        Args:
            batch: dependent on the backbone
        Returns:
            output of the configured head
        """
        raise NotImplementedError('Forward for wrapper has not been implemented.')
    
    
    def load_pretrained_backbone(
            self, 
    ):
        """  Based off of load checkpoint

        Args:
            weights_path: path to model checkpoint with only model weights
            ignore_modules: modules to ignore within the selected hierarchy.
                To ignore embedding related modules, set variable to ['patch_embedding', 'unembed']
            sel_prefix: '' selects all the modules within PrithviWxC. 'encoder.' selects encoder only.
            freeze: freezes the backbone when set.
            return_keys: If True, returns the output of load_state_dict(): missing_keys and unexpected_keys fields.

        Returns:
            If *return_keys* is True, `NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields.
        """
        if distributed.is_main_process():
            print(f"Loading pre-trained model weights {self.config.load_model}...")

        if self.config.load_exp:
            self.backbone, _, _, _, last_epoch, _, _ = self.checkpointer.load(model=self.backbone, optimizer=None, scheduler=None, dataloader=None, path=self.config.load_model)
            # try:
            #     self.backbone, _, _, _, last_epoch, _, _ = self.checkpointer.load(model=self.backbone, optimizer=None, scheduler=None, dataloader=None, path=self.config.load_model)
            #     start_epoch = last_epoch + 1
            # except Exception:
            #     print(f"Tried loading pretrained weights from {self.config.load_model}, but failed.")

        self.backbone = self.backbone.encoder


    def ignore_patch_embed(self, checkpoint, ignore_modules: Optional[list[str]] = None):
        """ Specifies PrithviWxC patch embedding layers and removes them from model checkpoint. """
        ignore_modules = [] if ignore_modules is None else ignore_modules
        ignore_layers = []
        for layer in checkpoint:
            if any(module in layer for module in ignore_modules):
                ignore_layers.append(layer)

        for layer in ignore_layers:
            checkpoint.pop(layer)

        return checkpoint, ignore_layers
    
    def freeze_unused_parameters(self, unused_parameters: list):
        for name, param in self.backbone.named_parameters():        
            if any(unused in name for unused in unused_parameters):
                param.requires_grad = False 

    def freeze_model(self, model: torch.nn.Module, ignore_layers: Optional[list] = None):
        """ Freeze given model for all layers expect the ones specified in ignore_layers. """
        ignore_layers = [] if ignore_layers is None else ignore_layers
        for name, param in model.named_parameters():        
            if name not in (ignore_layers):
                param.requires_grad = False


class ClimateDownscaleFinetuneModel(FinetuneWrapper):

    def __init__(
            self,
            embedding: torch.nn.Module,
            upscale: torch.nn.Module,
            backbone: torch.nn.Module,
            head: torch.nn.Module,
            embed_dim_backbone: int,
            n_lats_px_backbone: int,
            n_lons_px_backbone: int,
            patch_size_px_backbone: tuple[int, int],
            mask_unit_size_px_backbone: tuple[int, int],
            input_scalers_mu: torch.tensor,
            input_scalers_sigma: torch.tensor,
            input_scalers_epsilon: float,
            static_input_scalers_mu: torch.Tensor,
            static_input_scalers_sigma: torch.Tensor,
            static_input_scalers_epsilon: float,
            output_scalers_mu: torch.tensor,
            output_scalers_sigma: torch.tensor,
            n_input_timestamps: int = 1,
            embedding_static: Optional[torch.nn.Module] = None,
            n_bins: int = 512,
            return_logits: bool = False,
            residual: str = None,
            residual_connection: bool = False,
            config=None,
        ):
        """ Climate Downscaling Model based on pre-trained backbone. 
        Args:
            embedding: module used to embed input [C, H, W] -> [E, h, w]
            backbone: module that learns the system dynamics (optionally fully trained) [E, h, w] -> [E, h, w]
            head: module to shape output  [E, h, w] -> [O, H, W]
            n_lats_px_backbone: Total latitudes in data. In pixels.
            n_lons_px_backbone: Total longitudes in data. In pixels
            patch_size_px_backbone: Patch size for tokenization. In pixels lat/lon
            mask_unit_size_px_backbone: Size of each mask unit. In pixels lat/lon
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale input
            input_scalers_sigma:Tensor of size (in_channels,). Used to rescale input
            input_scalers_epsilon: Used to rescale input/ define a lower limit on std
            target_scalers_mu: Tensor of shape (in_channels,). Used to rescale output.
            target_scalers_sigma: Tensor of shape (in_channels,). Used to rescale output.
            n_bins: (optional) Used for cross entropy loss
            return_logits: (optional) Used to determine if we cross entropy loss
            residual: (optional) Indicates the residual mode of the model. for regression
                ['climate',  None]
            residual_connection: (optional) Use a skip/residual connection around the model backbone
        """

        super().__init__(backbone, head, config=config)

        self.n_input_timestamps = n_input_timestamps

        self.residual = residual if residual is not None else ''
        self.residual_connection = residual_connection

        self.embedding = embedding
        self.embedding_static = embedding_static

        self.upscale = upscale

        self.conv_after_backbone = nn.Conv2d(
            embed_dim_backbone, 
            embed_dim_backbone, 
            kernel_size=3, 
            stride=1,
            padding='same',
            padding_mode='replicate'
        )

        # Input shape [batch, time x parameter, lat, lon]
        self.input_scalers_mu = torch.nn.Parameter(
            input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_sigma = torch.nn.Parameter(
            input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.input_scalers_epsilon = input_scalers_epsilon

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_mu = nn.Parameter(
            static_input_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_sigma = nn.Parameter(
            static_input_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )
        self.static_input_scalers_epsilon = static_input_scalers_epsilon

        if output_scalers_mu is not None:
            self.output_scalers_mu = torch.nn.Parameter(
                output_scalers_mu.reshape(1, -1, 1, 1), requires_grad=False
            )
        self.output_scalers_sigma = torch.nn.Parameter(
            output_scalers_sigma.reshape(1, -1, 1, 1), requires_grad=False
        )

        assert n_lats_px_backbone % mask_unit_size_px_backbone[0] == 0
        assert n_lons_px_backbone % mask_unit_size_px_backbone[1] == 0
        assert mask_unit_size_px_backbone[0] % patch_size_px_backbone[0] == 0
        assert mask_unit_size_px_backbone[1] % patch_size_px_backbone[1] == 0

        self.local_shape_mu = (
            int(mask_unit_size_px_backbone[0] // patch_size_px_backbone[0]),
            int(mask_unit_size_px_backbone[1] // patch_size_px_backbone[1]),
        )
        self.global_shape_mu = (
            int(n_lats_px_backbone // mask_unit_size_px_backbone[0]),
            int(n_lons_px_backbone // mask_unit_size_px_backbone[1]),
        )
        self.patch_size_px = patch_size_px_backbone

        self.return_logits = return_logits
        if self.return_logits:
            self.to_logits = nn.Conv2d(
                in_channels=n_bins,
                out_channels=n_bins,
                kernel_size=1,
            )

    def swap_masking(self) -> None:
        return  

    #@profile
    def forward(self, batch: dict[str, torch.tensor]):
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', and 'static'.
                The associated torch tensors have the following shapes:
                x: Tensor of shape [batch, time x parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """

        B, _, H, W = batch['x'].shape
        # Scale inputs
        x_sep_time = batch['x'].view(B, self.n_input_timestamps, -1, H, W) # [batch, time x parameter, lat, lon] -> [batch, time, parameter, lat, lon]
        x_scale = (x_sep_time - self.input_scalers_mu.view(1, 1, -1, 1, 1)) / ( 
                self.input_scalers_sigma.view(1, 1, -1, 1, 1) + self.input_scalers_epsilon)
        x = x_scale.view(B, -1, H, W) # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        
        x_static = (batch['static_x'] - self.static_input_scalers_mu) / (
            self.static_input_scalers_sigma + self.static_input_scalers_epsilon
        )

        if self.residual == 'climate':
            # Scale climatology
            climate = (batch['climate_x'] - self.input_scalers_mu) / (
                self.input_scalers_sigma + self.input_scalers_epsilon
            )

            # concat with static in channels dimension
            x_static = torch.cat([x_static, climate], dim=1)

        # tokenization
        if self.embedding_static is None:
            x = torch.cat([x, x_static], dim=1) # combine the inputs and static in channel dimension
            x_shallow_feats = self.embedding(x)  # [batch, time x parameter, lat, lon] -> [batch, emb, lat*scale[0], lon*scale[0]]
        else:
            x_embedded = self.embedding(x) # [batch, time x parameter, lat, lon] -> [batch, emb, lat*scale[0], lon*scale[0]]
            static_embedded = self.embedding_static(x_static)
            x_shallow_feats = x_embedded + static_embedded

        x_upscale = self.upscale(x_shallow_feats)

        x_tokens = (
            x_upscale.reshape(
                B,
                -1,
                self.global_shape_mu[0],
                self.local_shape_mu[0],
                self.global_shape_mu[1],
                self.local_shape_mu[1],
            )
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3, 4)
            .flatten(1, 2)
        )  # [batch, embed, lat//patch_size, lon//patch_size] -> [batch, global seq, local seq, embed]

        x_deep_feats = self.backbone(x_tokens)  # [batch, global seq, local seq, embed]

        x_deep_feats = x_deep_feats.reshape(
            B,
            self.global_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            -1
        ).permute(0, 5, 1, 3, 2, 4)
        x_deep_feats = x_deep_feats.flatten(4, 5).flatten(2, 3)


        x_deep_feats = self.conv_after_backbone(x_deep_feats)

        if self.residual_connection:
            x = x_deep_feats + x_upscale

        x = self.head(x)  # [batch, out_channels, lat*scale[0]*scale[1], lon*scale[0]*scale[1]]

        if self.return_logits:
            x_out = self.to_logits(x)
        elif self.residual == 'climate':
            x_out = self.output_scalers_sigma * x + batch['climate_y']
        else:
            x_out = self.output_scalers_sigma * x + self.output_scalers_mu # [batch, 1, lat_high_res, lon_high_res]
        
        return x_out
