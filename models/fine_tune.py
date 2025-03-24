import sys
import os
import torch as t
import fire
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from models.hparams import setup_hparams
from models.make_models import make_vqvae, make_prior, make_up_model, restore_opt, save_checkpoint
from models.utils.logger import init_logging
from models.utils.audio_utils import audio_preprocess, audio_postprocess
from models.utils.torch_utils import zero_grad, count_parameters
from models.utils.dist_utils import isDist, use_cuda
from models.utils.ema import CPUEMA, FusedEMA, EMA
from models.utils.fp16 import FP16FusedAdam, FusedAdam, LossScalar, clipped_grad_scale, backward
from models.data.data_processor import DataProcessor
from AVSBenchDataset import AVSBenchDataset 

# Utility functions
def prepare_aud(x, hps):
    x = audio_postprocess(x.detach().contiguous(), hps)
    return allgather(x)

def log_aud(logger, tag, x, hps):
    logger.add_audios(tag, prepare_aud(x, hps), hps.sr, max_len=hps.max_len, max_log=hps.max_log)
    logger.flush()

def log_labels(logger, labeller, tag, y, hps):
    y = y.cpu().numpy()
    txt = ''
    for item in range(y.shape[0]):
        description = labeller.describe_label(y[item])
        artist, genre, lyrics = description['artist'], description['genre'], description['lyrics']
        txt += f'{item} artist:{artist}, genre:{genre}, lyrics:{lyrics}\n'
    logger.add_text(tag, txt)
    logger.flush()

def get_ddp(model, hps):
    rank = dist.get_rank()
    local_rank = rank % 8
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, bucket_cap_mb=hps.bucket)
    return ddp

def get_ema(model, hps):
    mu = hps.mu or (1. - (hps.bs * hps.ngpus/8.)/1000)
    ema = None
    if hps.ema and hps.train:
        if hps.cpu_ema:
            if dist.get_rank() == 0:
                print("Using CPU EMA")
            ema = CPUEMA(model.parameters(), mu=mu, freq=hps.cpu_ema_freq)
        elif hps.ema_fused:
            ema = FusedEMA(model.parameters(), mu=mu)
        else:
            ema = EMA(model.parameters(), mu=mu)
    return ema

def get_lr_scheduler(opt, hps):
    def lr_lambda(step):
        if hps.lr_use_linear_decay:
            lr_scale = hps.lr_scale * min(1.0, step / hps.lr_warmup)
            decay = max(0.0, 1.0 - max(0.0, step - hps.lr_start_linear_decay) / hps.lr_decay)
            if decay == 0.0:
                if dist.get_rank() == 0:
                    print("Reached end of training")
            return lr_scale * decay
        else:
            return hps.lr_scale * (hps.lr_gamma ** (step // hps.lr_decay)) * min(1.0, step / hps.lr_warmup)

    shd = t.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return shd

def get_optimizer(model, hps):
    # Optimizer
    betas = (hps.beta1, hps.beta2)
    if hps.fp16_opt:
        opt = FP16FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)
    else:
        opt = FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)

    # lr scheduler
    shd = get_lr_scheduler(opt, hps)

    restore_path = hps.restore_prior if hps.prior else hps.restore_vqvae
    restore_opt(opt, shd, restore_path)

    # fp16 dynamic loss scaler
    scalar = None
    if hps.fp16:
        rank = dist.get_rank()
        local_rank = rank % 8
        scalar = LossScalar(hps.fp16_loss_scale, scale_factor=2 ** (1./hps.fp16_scale_window))
        if local_rank == 0: print(scalar.__dict__)

    zero_grad(model)
    return opt, shd, scalar

def train(model, orig_model, opt, shd, scalar, ema, logger, metrics, data_processor, hps):
    model.train()
    orig_model.train()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd", gn="gn", g_l="gen_loss", p_l="prime_loss")
    else:
        _print_keys = dict(l="loss", sl="spectral_loss", rl="recons_loss", e="entropy", u="usage", uc="used_curr", gn="gn", pn="pn", dk="dk")

    for i, x in logger.get_range(data_processor.train_loader):
        if isinstance(x, (tuple, list)):
            x, y = x
        else:
            y = None
        if use_cuda:
            x = x.to('cuda', non_blocking=True)
            if y is not None:
                y = y.to('cuda', non_blocking=True)

        x_in = x = audio_preprocess(x, hps)
        log_input_output = (logger.iters % hps.save_iters == 0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output)
        else:
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

        # Forward
        x_out, loss, _metrics = model(x, **forw_kwargs)

        # Backward
        loss, scale, grad_norm, overflow_loss, overflow_grad = backward(loss=loss, params=list(model.parameters()),
                                                                         scalar=scalar, fp16=hps.fp16, logger=logger)
        grad_norm = allreduce(grad_norm, op=dist.ReduceOp.MAX)
        if overflow_loss or overflow_grad or grad_norm > hps.ignore_grad_norm > 0:
            zero_grad(orig_model)
            continue

        # Step opt
        logger.step()
        opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        zero_grad(orig_model)
        lr = hps.lr if shd is None else shd.get_lr()[0]
        if shd is not None: shd.step()
        if ema is not None: ema.step()

        # Logging
        for key, val in _metrics.items():
            _metrics[key] = val.item()
        _metrics["loss"] = loss = loss.item() * hps.iters_before_update  # Free graph
        _metrics["gn"] = grad_norm
        _metrics["lr"] = lr

        for key, val in _metrics.items():
            _metrics[key] = metrics.update(key, val, x.shape[0])
            if logger.iters % hps.log_steps == 0:
                logger.add_scalar(key, _metrics[key])

        # Save checkpoint
        if hps.save and (logger.iters % hps.save_iters == 1):
            if ema is not None: ema.swap()
            orig_model.eval()
            name = f"step_{logger.iters}"
            save_checkpoint(logger, name, orig_model, opt, dict(step=logger.iters), hps)
            orig_model.train()
            if ema is not None: ema.swap()

        logger.set_postfix(**{print_key:_metrics[key] for print_key, key in _print_keys.items()})

    return {key: metrics.avg(key) for key in _metrics.keys()}

def run(hps="teeny", port=29500, **kwargs):
    from models.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs

    # Setup dataset
    image_dir = 'path_to_images'
    audio_dir = 'path_to_audio'
    dataset = AVSBenchDataset(root_dir='path_to_data', use_mel=True)
    data_processor = DataProcessor(hps, dataset)

    # Setup models
    vqvae = make_vqvae(hps, device)
    print(f"Parameters VQVAE: {count_parameters(vqvae)}")
    if hps.prior:
        prior = make_prior(hps, vqvae, device)
        print(f"Parameters Prior: {count_parameters(prior)}")
        model = prior
    else:
        model = vqvae

    # Setup optimizer, EMA, and distributed model
    opt, shd, scalar = get_optimizer(model, hps)
    ema = get_ema(model, hps)
    if isDist:
        distributed_model = get_ddp(model, hps)
    else:
        distributed_model = model

    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Run training, eval, sample
    for epoch in range(hps.curr_epoch, hps.epochs):
        metrics.reset()
        data_processor.set_epoch(epoch)
        if hps.train:
            train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, data_processor, hps)
            train_metrics['epoch'] = epoch
            if rank == 0:
                print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
            dist.barrier()

        if hps.test:
            if ema: ema.swap()
            test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
            test_metrics['epoch'] = epoch
            if rank == 0:
                print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
            dist.barrier()
            if ema: ema.swap()
        dist.barrier()

if __name__ == '__main__':
    fire.Fire(run)

