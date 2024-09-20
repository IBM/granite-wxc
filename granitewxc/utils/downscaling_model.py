from typing import Optional
from logging import Logger

import numpy as np
import torch

from PrithviWxC.model import PrithviWxCEncoderDecoder, SWINShift

from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils import distributed
from granitewxc.utils.data import (assemble_input_scalers,
                                       assemble_target_scalers,
                                       assemble_output_scalers,
                                       assemble_static_input_scalers,
                                       assemble_climate_scalers)
from granitewxc.models.finetune_model import PatchEmbed, ClimateDownscaleFinetuneModel
from granitewxc.decoders.downscaling import ConvEncoderDecoder


def get_scalers(config: ExperimentConfig):
    """
    calls assemble scalers func. depending on the dataset
    """
    # Input and target scalers
    if config.data.type == 'merra2':

        input_mu, input_sigma = assemble_input_scalers(config)

        if config.model.__dict__.get('residual', '') == 'climate':
            target_variance = assemble_climate_scalers(config)
            target_sigma = torch.sqrt(target_variance)
            target_mu = None
        else:
            target_mu, target_sigma = assemble_target_scalers(config)

        input_static_mu, input_static_sigma = assemble_static_input_scalers(config)

    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')

    return dict(
        input_mu=input_mu,
        input_sigma=input_sigma,
        input_static_mu=input_static_mu,
        input_static_sigma=input_static_sigma,
        target_mu=target_mu,
        target_sigma=target_sigma,
    )


def get_merra2_embedding_module(config: ExperimentConfig,):
    """
    Initializes the object required to embed merra2 data
    """

    n_parameters = (len(config.data.input_surface_vars) + len(config.data.input_levels) * len(
        config.data.input_vertical_vars))


    patch_embedding = PatchEmbed(
        patch_size=config.model.downscaling_patch_size,
        channels=n_parameters * config.data.n_input_timestamps,
        embed_dim=config.model.downscaling_embed_dim,
    )

    n_static_parameters = config.model.num_static_channels + len(config.data.input_static_surface_vars)
    if config.model.residual == 'climate':
        n_static_parameters += n_parameters

    patch_embedding_static = PatchEmbed(
        patch_size=config.model.downscaling_patch_size,
        channels=n_static_parameters,
        embed_dim=config.model.downscaling_embed_dim,
    )

    return patch_embedding, patch_embedding_static


def get_merra2_upscaling_module(config: ExperimentConfig,):
    """
    Initializes the object required to embed merra2 data
    """

    n_parameters = (len(config.data.input_surface_vars) + len(config.data.input_levels) * len(
        config.data.input_vertical_vars))


    patch_embedding = ConvEncoderDecoder(
        in_channels=n_parameters * config.data.n_input_timestamps,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=config.model.embed_dim,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[0],
        scale=config.model.encoder_decoder_scale_per_stage[0],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )

    n_static_parameters = config.model.num_static_channels + len(config.data.input_static_surface_vars)
    if config.model.residual == 'climate':
        n_static_parameters += n_parameters

    patch_embedding_static = ConvEncoderDecoder(
        in_channels=n_static_parameters,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=config.model.embed_dim,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[0],
        scale=config.model.encoder_decoder_scale_per_stage[0],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )

    return patch_embedding, patch_embedding_static


def get_backbone(config) -> PrithviWxCEncoderDecoder:

    print("Encoder shifting: {}".format(config.model.encoder_shift))

    if config.model.encoder_shift:
        n_lats_px=int(config.data.input_size_lat * np.prod(config.model.encoder_decoder_scale_per_stage[0]))
        n_lons_px=int(config.data.input_size_lon * np.prod(config.model.encoder_decoder_scale_per_stage[0]))
        local_shape_mu = (
            config.mask_unit_size[0] // config.model.token_size[0],
            config.mask_unit_size[1] // config.model.token_size[1],
        )
        global_shape_mu = (
            n_lats_px // config.mask_unit_size[0],
            n_lons_px // config.mask_unit_size[1],
        )
        e_shifter = SWINShift(
            config.mask_unit_size,
            global_shape_mu,
            local_shape_mu,
            config.model.token_size,
            n_context_tokens=0,
        )
    else:
        e_shifter = None

    return PrithviWxCEncoderDecoder(
        embed_dim=config.model.embed_dim,
        n_blocks=config.model.n_blocks_encoder,
        mlp_multiplier=config.model.mlp_multiplier,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout_rate,
        drop_path=config.model.drop_path,
        shifter=e_shifter,
    )


def get_finetune_model(config: ExperimentConfig, logger: Optional[Logger] = None) -> torch.nn.Module:
    """
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
        logger: wandb Run object as returned by wandb.init.
    Returns:
        The configured model.
    """

    if distributed.is_main_process():
        print("Creating the model.")

    #########################################################
    # 0. Setup parameters/scalers
    #########################################################
    # set default kernel size
    if 'encoder_decoder_kernel_size_per_stage' not in config.model.__dict__:
        config.model.encoder_decoder_kernel_size_per_stage = [[3]*len(inner) for inner in config.model.encoder_decoder_scale_per_stage]

    n_output_parameters = len(config.data.output_vars)
    if config.model.__dict__.get('loss_type', 'patch_rmse_loss')=='cross_entropy':
        if config.model.__dict__.get('cross_entropy_bin_width_type', 'uniform') == 'uniform':
            n_output_parameters = config.model.__dict__.get('cross_entropy_n_bins', 512)
        else:
            n_output_parameters = len(np.load(config.model.cross_entropy_bin_boundaries_file)) + 1

    scalers = get_scalers(config)

    #########################################################
    # 1. Patch Embedding/Shallow Feature Extraction
    #########################################################
    if config.data.type == 'merra2':  # merra2
        embedding, embedding_static = get_merra2_embedding_module(config)
    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')

    #########################################################
    # 2. Upscale before FM
    # Keep token resolution similar to trained backbone
    #########################################################
    upscale = ConvEncoderDecoder(
        in_channels=config.model.downscaling_embed_dim,
        channels=config.model.encoder_decoder_conv_channels,
        out_channels=config.model.embed_dim,
        kernel_size=config.model.encoder_decoder_kernel_size_per_stage[0],
        scale=config.model.encoder_decoder_scale_per_stage[0],
        upsampling_mode=config.model.encoder_decoder_upsampling_mode,
    )


    #########################################################
    # 3. FM/Deep Feature Extraction
    #########################################################

    backbone = get_backbone(config)

    #########################################################
    # 4. Upscale after FM
    #########################################################
    if config.model.encoder_decoder_type == 'conv':
        head = ConvEncoderDecoder(
                in_channels=config.model.embed_dim,
                channels=config.model.encoder_decoder_conv_channels,
                out_channels=n_output_parameters,
                kernel_size=config.model.encoder_decoder_kernel_size_per_stage[1],
                scale=config.model.encoder_decoder_scale_per_stage[1],
                upsampling_mode=config.model.encoder_decoder_upsampling_mode,
        )
    else:
        raise NotImplementedError(f"Head type {config.model.encoder_decoder_type} not implemented.")

    #########################################################
    # 5. Putting it all together
    #########################################################
    model = ClimateDownscaleFinetuneModel(
        embedding=embedding,
        embedding_static=embedding_static,
        upscale=upscale,
        backbone=backbone,
        head=head,
        input_scalers_mu=scalers['input_mu'],
        input_scalers_sigma=scalers['input_sigma'],
        input_scalers_epsilon=1e-6,
        static_input_scalers_mu=scalers['input_static_mu'],
        static_input_scalers_sigma=scalers['input_static_sigma'],
        static_input_scalers_epsilon=1e-6,
        output_scalers_mu=scalers['target_mu'],
        output_scalers_sigma=scalers['target_sigma'],
        n_input_timestamps=config.data.n_input_timestamps,
        embed_dim_backbone=config.model.embed_dim,
        n_lats_px_backbone=int(config.data.input_size_lat * np.prod(config.model.encoder_decoder_scale_per_stage[0])),
        n_lons_px_backbone=int(config.data.input_size_lon * np.prod(config.model.encoder_decoder_scale_per_stage[0])),
        patch_size_px_backbone=config.model.token_size,
        mask_unit_size_px_backbone=config.mask_unit_size,
        n_bins=n_output_parameters,
        return_logits=config.model.__dict__.get('loss_type')=='cross_entropy',
        residual=config.model.__dict__.get('residual', None),
        residual_connection=config.model.__dict__.get('residual_connection', False),
        config=config,
    )

    if distributed.is_main_process():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if logger is not None:
            logger.info(f"--> model has {total_params:,.0f} params.")
    return model
