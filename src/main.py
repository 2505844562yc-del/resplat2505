import os
from pathlib import Path
import warnings
import copy

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt, no_resume_upsampler
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    if cfg_dict["mode"] == "train" and cfg_dict["train"]["eval_model_every_n_val"] > 0:
        eval_cfg_dict = copy.deepcopy(cfg_dict)
        dataset_dir = str(cfg_dict["dataset"]["roots"]).lower()
        if "re10k" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/evaluation_index_re10k.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 4:
                eval_path = "assets/re10k_start_0_distance_150_ctx_4v_tgt_6v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/re10k_start_0_distance_200_ctx_6v_tgt_6v.json"
            else:
                if cfg_dict["trainer"]["eval_index"] is not None:
                    eval_path = None  # placeholder
                else:
                    raise ValueError("unsupported number of views for re10k")
        elif "dl3dv" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/dl3dv_start_0_distance_20_ctx_2v_tgt_4v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 8:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 16:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_80_ctx_16v_tgt_16v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 32:
                eval_path = "assets/dl3dv_evaluation/dl3dv_start_0_distance_160_ctx_32v_tgt_24v.json"
            elif cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 64:
                eval_path = "assets/dl3dv_benchmark/dl3dv_ctx_64v_tgt_every8th.json"
            else:
                eval_path = None
                # raise ValueError("unsupported number of views for dl3dv")
        elif "scannet" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = "assets/evaluation_index_scannet_view2.json"
            else:
                raise ValueError("unsupported number of views for scannet")
        elif "tartanair" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 2:
                eval_path = 'assets/evaluation_index_tartanair_view2.json'
            else:
                raise ValueError("unsupported number of views for tartanair")
        else:
            raise Exception("Fail to load eval index path")
        eval_cfg_dict["dataset"]["view_sampler"] = {
            "name": "evaluation",
            "index_path": eval_path,
            "num_context_views": cfg_dict["dataset"]["view_sampler"]["num_context_views"],
        }

        # specify eval index
        if cfg_dict["trainer"]["eval_index"] is not None:
            eval_cfg_dict["dataset"]["view_sampler"]["index_path"] = cfg_dict["trainer"]["eval_index"]

        assert eval_cfg_dict["dataset"]["view_sampler"]["index_path"] is not None, "no evaluation index path found!"

        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}"))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled" and cfg.mode == "train":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        run_name = os.path.basename(cfg_dict.output_dir)
        if cfg_dict.log_slurm_id:
            run_name += f" ({os.environ.get('SLURM_JOB_ID')})"
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=run_name,
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    if cfg.checkpointing.resume:
        if not os.path.exists(output_dir / 'checkpoints'):
            checkpoint_path = None
        else:
            checkpoint_path = find_latest_ckpt(output_dir / 'checkpoints')
            print(f'resume from {checkpoint_path}')
    else:
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    dist_strategy = 'ddp'

    if cfg.model.encoder.use_checkpointing or cfg.model.encoder.init_use_checkpointing:
        # need this for recurrent update or init pt model
        dist_strategy = DDPStrategy(static_graph=True)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=torch.cuda.device_count(),
        strategy=dist_strategy if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=(
            None if eval_cfg is None else eval_cfg.dataset
        ),
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))

    strict_load = not cfg.checkpointing.no_strict_load

    if cfg.mode == "train":
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')

            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            if 'model' in pretrained_model:
                pretrained_model = pretrained_model['model']

            if cfg.checkpointing.no_resume_upsampler:
                pretrained_model = no_resume_upsampler(pretrained_model)
                strict_load = False

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )

        # load pretrained update module
        if cfg.checkpointing.resume_update_module is not None:
            pretrained_model = torch.load(cfg.checkpointing.resume_update_module, map_location='cpu')

            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            if 'model' in pretrained_model:
                pretrained_model = pretrained_model['model']

            # Filter and load only matching "update_" parameters
            filtered_dict = {
                k: v for k, v in pretrained_model.items()
                if "encoder.update" in k and k in model_wrapper.state_dict() and v.shape == model_wrapper.state_dict()[k].shape
            }

            # Load them using strict=False so it skips missing/unmatched keys
            model_wrapper.load_state_dict(filtered_dict, strict=False)

            print(
                cyan(
                    f"Loaded pretrained update module: {cfg.checkpointing.resume_update_module}"
                )
            )

        if cfg.model.encoder.num_refine > 0:
            print('train refine only')
            for name, params in model_wrapper.named_parameters():
                trainable = 'encoder.update' in name
                if cfg.model.encoder.use_semantic_gaussian_init:
                    trainable = trainable or 'encoder.semantic_init' in name
                if not trainable:
                    params.requires_grad = False

        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth model only
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )

        # load pretrained update module
        if cfg.checkpointing.resume_update_module is not None:
            pretrained_model = torch.load(cfg.checkpointing.resume_update_module, map_location='cpu')

            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            if 'model' in pretrained_model:
                pretrained_model = pretrained_model['model']

            # Filter and load only matching "update_" parameters
            filtered_dict = {
                k: v for k, v in pretrained_model.items()
                if "encoder.update" in k and k in model_wrapper.state_dict() and v.shape == model_wrapper.state_dict()[k].shape
            }

            # Load them using strict=False so it skips missing/unmatched keys
            model_wrapper.load_state_dict(filtered_dict, strict=False)

            print(
                cyan(
                    f"Loaded pretrained update module: {cfg.checkpointing.resume_update_module}"
                )
            )
            
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
