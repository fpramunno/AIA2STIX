# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:47:31 2025

@author: pio-r
"""

# import debugpy

# debugpy.connect(("v000675", 5678))  # VS Code listens on login node
# print("✅ Connected to VS Code debugger!")
# debugpy.wait_for_client()
# print("🎯 Debugger attached!")

def main():

    import argparse
    import os
    from copy import deepcopy
    import json
    from pathlib import Path
    import time
                
    import matplotlib.pyplot as plt
    import imageio
    import io
    import accelerate
    import safetensors.torch as safetorch
    import torch
    import torch._dynamo
    from torch import optim
    from tqdm.auto import tqdm
    import numpy as np
    import src as K

    from util import generate_samples
    import torch
    from src.data.dataset import get_aia2stix_data_objects

    def chi_square_distance(pred_vis, true_vis):
        """Calculate chi-square distance between predicted and true visibilities."""
        # Flatten to (24*2,) for easier computation
        pred_flat = pred_vis.view(-1).cpu().numpy()
        true_flat = true_vis.view(-1).cpu().numpy()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        chi_sq = np.sum((pred_flat - true_flat)**2 / (np.abs(true_flat) + epsilon))
        return chi_sq

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=32,
                help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                help='compile the model')
    p.add_argument('--config', type=str, required=True,
                help='the configuration file')
    p.add_argument('--data-path', type=str, default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                help='the path of the dataset')
    p.add_argument('--saving-path', type=str, default="/mnt/nas05/data01/francesco/AIA2STIX/", 
                help='the path where to save the model')
    p.add_argument('--dir-name', type=str, default='aia_2_stix_v1',
                help='the directory name to use')  # <---- Added this line
    p.add_argument('--end-step', type=int, default=None,
                help='the step to end training at')
    p.add_argument('--max-epochs', type=int, default=None,
                help='the maximum number of epochs to train for')
    p.add_argument('--evaluate-every', type=int, default=5,
                help='How often to evaluate the model in epochs') 
    p.add_argument('--evaluate-only', action='store_true',
                help='evaluate instead of training')
    p.add_argument('--gns', action='store_true',
                help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                help='reset the EMA')
    p.add_argument('--resume', type=str,
                help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                help='the inference checkpoint to resume from')
    p.add_argument('--save-every', type=int, default=5,
                help='save every this many epochs')
    p.add_argument('--seed', type=int,
                help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                choices=['fork', 'forkserver', 'spawn'],
                help='the multiprocessing start method')
    p.add_argument('--use_wandb', type=bool, default=True,
                help='Use wandb')

    p.add_argument('--wandb-entity', type=str,
                help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                help='save model to wandb')
    p.add_argument('--wandb-run-name', type=str,
                help='the wandb run name')

    args = p.parse_args()

    dir_path_res = os.path.join(args.saving_path, f"results_{args.dir_name}")
    dir_path_mdl = os.path.join(args.saving_path, f"model_{args.dir_name}")

    if not os.path.exists(dir_path_res):
        os.makedirs(dir_path_res, exist_ok=True)
        
    if not os.path.exists(dir_path_mdl):
        os.makedirs(dir_path_mdl, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']

    assert len(model_config['input_size']) == 2 # and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps,
                                            mixed_precision=args.mixed_precision, 
                kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(find_unused_parameters=True)])

    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    # Model definition,
    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    if args.compile:
        inner_model.compile()
        inner_model_ema.compile()

    if accelerator.is_main_process:
        # Detailed model parameter analysis
        total_params = sum(p.numel() for p in inner_model.parameters())
        trainable_params = sum(p.numel() for p in inner_model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("=" * 60)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f'Total parameters:      {total_params:,}')
        print(f'Trainable parameters:  {trainable_params:,}')
        print(f'Non-trainable params:  {non_trainable_params:,}')
        print(f'Trainable percentage:  {100 * trainable_params / total_params:.2f}%')
        
        # Memory estimation (assuming float32)
        param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        print(f'Model size (approx):   {param_size_mb:.2f} MB')
        
        # Parameter breakdown by layer type
        param_breakdown = {}
        for name, param in inner_model.named_parameters():
            layer_type = name.split('.')[0] if '.' in name else name
            if layer_type not in param_breakdown:
                param_breakdown[layer_type] = 0
            param_breakdown[layer_type] += param.numel()
        
        print("\nParameter breakdown by layer:")
        for layer_type, count in sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total_params
            print(f'  {layer_type:20}: {count:>10,} ({percentage:5.2f}%)')
        
        print("=" * 60)


    # WANDB LOGGING
    use_wandb = args.use_wandb # accelerator.is_main_process and args.wandb_project
    if accelerator.is_main_process and use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(project="aia_2_stix", entity="francescopio", name=args.wandb_run_name, config=log_config, save_code=True)

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(groups,
                        lr=lr,
                        betas=tuple(opt_config['betas']),
                        eps=opt_config['eps'],
                        weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(groups,
                                lr=lr,
                                betas=tuple(opt_config['betas']),
                                eps=opt_config['eps'],
                                weight_decay=opt_config['weight_decay'])
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(groups,
                        lr=lr,
                        momentum=opt_config.get('momentum', 0.),
                        nesterov=opt_config.get('nesterov', False),
                        weight_decay=opt_config.get('weight_decay', 0.))
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                inv_gamma=sched_config['inv_gamma'],
                                power=sched_config['power'],
                                warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                    num_steps=sched_config['num_steps'],
                                    decay=sched_config['decay'],
                                    warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                max_value=ema_sched_config['max_value'])
    ema_stats = {}

    # Load the dataset

    train_dataset, train_sampler, train_dl = get_aia2stix_data_objects(
        vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='train',
        seed=42,
        enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data/"
    )

    val_dataset, val_sampler, val_dl = get_aia2stix_data_objects(
        vis_path="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
        data_path="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
        batch_size=args.batch_size,
        distributed=False,
        num_data_workers=args.num_workers,
        split='valid',
        seed=42,
        enc_data_path="/mnt/nas05/astrodata01/aia_2_stix/encoded_data/"
    )

    print('Train loader and Valid loader are up!')

    # Prepare the model, optimizer, and dataloaders with the accelerator
    inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)

    use_wandb = args.use_wandb

    if use_wandb and accelerator.is_main_process:
        wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    # Define the model 
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = Path(f'{args.name}_state_{args.dir_name}.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                    max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt

    def save():
        accelerator.wait_for_everyone()
        filename = os.path.join(args.saving_path, dir_path_mdl, f"{args.name}_epoch_{epoch:04d}.pth") 
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    losses_since_last_print = []
    model = model.to(device)
    try:
        while True:
            # Training Loop
            epoch_train_loss = 0  # Track total training loss
            num_train_batches = len(train_dl)  # Number of batches
            model.train()
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()
                
                print('Here')
                with accelerator.accumulate(model):
                    # Track memory before processing
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_before = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory before processing: Allocated: {mem_before:.2f} GB, Reserved: {reserved_before:.2f} GB")

                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    target_vis = batch[1].to(device, non_blocking=True).reshape(inpt.shape[0], 1, 24, 2)
                    enc_inpt = batch[2].to(device, non_blocking=True).reshape(inpt.shape[0], 1, 24, 2)

                    extra_args = {}
                    noise = torch.randn_like(target_vis).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([target_vis.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(target_vis, enc_inpt, noise, sigma, mapping_cond=None, **extra_args)

                    # Evita NCCL timeout: non fare gather durante il training!
                    loss = losses.mean().item()
                    losses_since_last_print.append(loss)
                    epoch_train_loss += loss  # Accumulate loss

                    # Backward pass
                    accelerator.backward(losses.mean())

                    # Track memory after backward pass
                    torch.cuda.synchronize()
                    mem_after_backward = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_after_backward = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory after backward pass: Allocated: {mem_after_backward:.2f} GB, Reserved: {reserved_after_backward:.2f} GB")

                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, inpt.shape[0], inpt.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    # Track memory after optimizer step
                    torch.cuda.synchronize()
                    mem_after_step = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB
                    reserved_after_step = torch.cuda.memory_reserved(device) / (1024 ** 3)  # Convert to GB
                    print(f"Memory after optimizer step: Allocated: {mem_after_step:.2f} GB, Reserved: {reserved_after_step:.2f} GB")

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                step += 1

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return
            
            epoch_train_loss /= num_train_batches 

            # **Validation Loop (After Training, Before wandb Logging)**
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc="Validation", disable=not accelerator.is_main_process):
                    inpt = batch[0].contiguous().float().to(device, non_blocking=True)
                    target_vis = batch[1].to(device, non_blocking=True).reshape(inpt.shape[0], 1, 24, 2)
                    enc_inpt = batch[2].to(device, non_blocking=True).reshape(inpt.shape[0], 1, 24, 2)

                    extra_args = {}
                    noise = torch.randn_like(target_vis).to(device)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([target_vis.shape[0]], device=device)

                    with K.models.checkpointing(args.checkpointing):
                        losses = model.loss(target_vis, enc_inpt, noise, sigma, mapping_cond=None, **extra_args)

                    # Make sure we only gather scalar loss (not batch tensor)
                    loss_value = losses.mean().detach()
                    gathered_loss = accelerator.gather_for_metrics(loss_value)

                    # Accumulate average across ranks only from main process
                    if accelerator.is_main_process:
                        val_loss += gathered_loss.mean().item()

            # Final averaging
            if accelerator.is_main_process:
                val_loss /= len(val_dl)

            # Print validation loss
            if accelerator.is_main_process:
                tqdm.write(f"Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Validation Loss: {val_loss:.6f}")

            # Sampling and Chi-square evaluation
            if epoch % args.evaluate_every == 0 and accelerator.is_main_process:
                
                # Get a batch for sampling
                test_batch = next(iter(val_dl))
                test_inpt = test_batch[0][:1].contiguous().float().to(device)  # Take first sample
                test_target_vis = test_batch[1][:1].to(device).reshape(1, 1, 24, 2)  # Original visibility
                test_enc_inpt = test_batch[2][:1].to(device).reshape(1, 1, 24, 2)  # Encoded input
                
                # Generate samples using the model
                # Use encoded input as conditioning
                samples = generate_samples(
                    model_ema, 
                    1, 
                    device, 
                    cond_label=None, 
                    sampler="dpmpp_2m_sde", 
                    cond_img=test_enc_inpt
                )
                
                # Extract predicted visibilities
                predicted_vis = samples[0]  # Shape: (1, 1, 24, 2)
                
                # Calculate chi-square distance
                
                chi_sq_dist = chi_square_distance(predicted_vis, test_target_vis)
                
                # Print results
                tqdm.write(f"Epoch {epoch}: Chi-square distance = {chi_sq_dist:.6f}")
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Convert to numpy for plotting
                pred_vis_np = predicted_vis[0].cpu().numpy()      # [0] gives [24, 2]
                true_vis_np = test_target_vis[0, 0].cpu().numpy()   # Shape: (24, 2)
                
                # Plot real parts
                axes[0, 0].plot(pred_vis_np[:, 0], 'b-', label='Predicted Real', linewidth=2)
                axes[0, 0].plot(true_vis_np[:, 0], 'r--', label='True Real', linewidth=2)
                axes[0, 0].set_title('Real Part Comparison')
                axes[0, 0].set_xlabel('Visibility Index')
                axes[0, 0].set_ylabel('Real Value')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot imaginary parts
                axes[0, 1].plot(pred_vis_np[:, 1], 'b-', label='Predicted Imag', linewidth=2)
                axes[0, 1].plot(true_vis_np[:, 1], 'r--', label='True Imag', linewidth=2)
                axes[0, 1].set_title('Imaginary Part Comparison')
                axes[0, 1].set_xlabel('Visibility Index')
                axes[0, 1].set_ylabel('Imaginary Value')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot amplitude comparison
                pred_amp = np.sqrt(pred_vis_np[:, 0]**2 + pred_vis_np[:, 1]**2)
                true_amp = np.sqrt(true_vis_np[:, 0]**2 + true_vis_np[:, 1]**2)
                axes[1, 0].plot(pred_amp, 'b-', label='Predicted Amplitude', linewidth=2)
                axes[1, 0].plot(true_amp, 'r--', label='True Amplitude', linewidth=2)
                axes[1, 0].set_title('Amplitude Comparison')
                axes[1, 0].set_xlabel('Visibility Index')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot phase comparison
                pred_phase = np.arctan2(pred_vis_np[:, 1], pred_vis_np[:, 0])
                true_phase = np.arctan2(true_vis_np[:, 1], true_vis_np[:, 0])
                axes[1, 1].plot(pred_phase, 'b-', label='Predicted Phase', linewidth=2)
                axes[1, 1].plot(true_phase, 'r--', label='True Phase', linewidth=2)
                axes[1, 1].set_title('Phase Comparison')
                axes[1, 1].set_xlabel('Visibility Index')
                axes[1, 1].set_ylabel('Phase (radians)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add overall title with chi-square distance
                fig.suptitle(f'Visibility Comparison - Epoch {epoch} - Chi-square Distance: {chi_sq_dist:.6f}', 
                            fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(dir_path_res, f'visibility_comparison_epoch_{epoch}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                
                print(f"Visibility comparison plot saved to: {plot_path}")
            
            # **wandb Logging (Now Includes Validation Loss)**
            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'lr': sched.get_last_lr()[0],
                    'ema_decay': ema_decay,
                    'Sampled images': wandb.Image(plt)
                }
                if args.gns:
                    log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                
                wandb.log(log_dict)
                plt.close()
            
            # Save model every N epochs instead of every epoch
            if epoch % args.save_every == 0:
                save()
                
            epoch += 1  # Move to the next epoch
            
            # Check if we've reached the maximum number of epochs
            if args.max_epochs is not None and epoch >= args.max_epochs:
                if accelerator.is_main_process:
                    tqdm.write(f'Reached maximum epochs ({args.max_epochs}). Training complete!')
                return

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()