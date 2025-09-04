from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import shutil
import threading
import time
import urllib
import warnings
import numpy as np

from PIL import Image
import safetensors
import torch
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import functional as TF

from torchvision.transforms import Resize, Normalize
from torchvision.transforms import Compose
import math

CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "Bx": {"min": -250, "max": 250, "scaling": None},
    "By": {"min": -250, "max": 250, "scaling": None},
    "Bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=256, channel="171", mask_limb=False, radius_scale_factor=1.0):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.

    Apply the normalization necessary for the SDO ML Dataset. Depending on the channel, it:
      - masks the limb with 0s
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 256.
        channel (str, optional): [The SDO channel]. Defaults to 171.
        mask_limb (bool, optional): [Whether to mask the limb]. Defaults to False.
        radius_scale_factor (float, optional): [Allows to scale the radius that is used for masking the limb]. Defaults to 1.0.
    Returns:
        [Transform]
    """

    transforms = []

    # also refer to
    # https://pytorch.org/vision/stable/transforms.html
    # https://github.com/i4Ds/SDOBenchmark/blob/master/dataset/data/load.py#L363
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO does it make sense to use vflip(x) in order to align the solar North as in JHelioviewer?
        # otherwise this has to be done during inference
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    elif preprocess_config["scaling"] == "sqrt":
        def lambda_transform(x): return torch.sqrt(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.sqrt(preprocess_config["min"])
        std = math.sqrt(preprocess_config["max"]) - \
            math.sqrt(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    def limb_mask_transform(x):
        h, w = x.shape[1], x.shape[2]  # C x H x W

        # fixed disk size of Rs of 976 arcsec, pixel size in the scaled image (512x512) is ~4.8 arcsec
        original_resolution = 4096
        scaled_resolution = h
        pixel_size_original = 0.6
        radius_arcsec = 976.0
        radius = (radius_arcsec / pixel_size_original) / \
            original_resolution * scaled_resolution

        mask = create_circular_mask(
            h, w, radius=radius*radius_scale_factor)
        mask = torch.as_tensor(mask, device=x.device)
        return torch.where(mask, x, torch.tensor(0.0))

    if mask_limb:
        def mask_lambda_func(x):
            return limb_mask_transform(x)
        transforms.append(mask_lambda_func)
        # transforms.append(Lambda(lambda x: limb_mask_transform(x)))

    transforms.append(Resize((target_size, target_size)))
    # TODO find out if these transforms make sense
    def test_lambda_func(x):
        return lambda_transform(x)
    transforms.append(test_lambda_func)
    # transforms.append(Lambda(lambda x: lambda_transform(x)))
    transforms.append(Normalize(mean=[mean], std=[std]))
    # required to remove strange distribution of pixels (everything too bright)
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    return Compose(transforms)

def create_circular_mask(h, w, center=None, radius=None):
    # TODO investigate the use of a circular mask to prevent focussing to much on the limb
    # https://gitlab.com/jdonzallaz/solarnet-app/-/blob/master/src/prediction.py#L9

    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def get_alpha(vis_arr):
    # Compute alpha for a single vis array
    norm_vis = np.sqrt(np.square(vis_arr[:, 0]) + np.square(vis_arr[:, 1]))
    alpha = (0.5) * np.max(norm_vis) # / (pix_size * pix_size)
    return alpha

def from_pil_image(x):
    """Converts from a PIL image to a tensor."""
    x = TF.to_tensor(x)
    if x.ndim == 2:
        x = x[..., None]
    return x * 2 - 1


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def hf_datasets_augs_helper(examples, transform, image_key, mode='RGB'):
    """Apply passed in transforms for HuggingFace Datasets."""
    images = [transform(image.convert(mode)) for image in examples[image_key]]
    return {image_key: images}


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """Downloads a file if it does not exist, optionally checking its SHA-256 hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, 'wb') as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if digest != file_digest:
            raise OSError(f'hash of {path} (url: {url}) failed to validate')
    return path


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


class InverseLR(optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """Implements an exponential learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr. Decays the learning rate
    continuously by decay (default 0.5) every num_steps steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (float): The number of steps to decay the learning rate by decay in.
        decay (float): The factor by which to decay the learning rate every num_steps
            steps. Default: 0.5.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, num_steps, decay=0.5, warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.num_steps = num_steps
        self.decay = decay
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ConstantLRWithWarmup(optim.lr_scheduler._LRScheduler):
    """Implements a constant learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup=0., last_epoch=-1, verbose=False):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        return [warmup * base_lr for base_lr in self.base_lrs]


def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


stratified_settings = threading.local()


@contextmanager
def enable_stratified(group=0, groups=1, disable=False):
    """A context manager that enables stratified sampling."""
    try:
        stratified_settings.disable = disable
        stratified_settings.group = group
        stratified_settings.groups = groups
        yield
    finally:
        del stratified_settings.disable
        del stratified_settings.group
        del stratified_settings.groups


@contextmanager
def enable_stratified_accelerate(accelerator, disable=False):
    """A context manager that enables stratified sampling, distributing the strata across
    all processes and gradient accumulation steps using settings from Hugging Face Accelerate."""
    try:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        acc_steps = accelerator.gradient_state.num_steps
        acc_step = accelerator.step % acc_steps
        group = rank * acc_steps + acc_step
        groups = world_size * acc_steps
        with enable_stratified(group, groups, disable=disable):
            yield
    finally:
        pass


def stratified_with_settings(shape, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution, using settings from a context
    manager."""
    if not hasattr(stratified_settings, 'disable') or stratified_settings.disable:
        return torch.rand(shape, dtype=dtype, device=device)
    return stratified_uniform(
        shape, stratified_settings.group, stratified_settings.groups, dtype=dtype, device=device
    )


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = stratified_with_settings(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (stratified_with_settings(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_with_settings(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_split_log_normal(shape, loc, scale_1, scale_2, device='cpu', dtype=torch.float32):
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


class FolderOfImages(data.Dataset):
    """Recursively finds all images in a directory. It does not support
    classes/targets."""

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(path for path in self.root.rglob('*') if path.suffix.lower() in self.IMG_EXTENSIONS)

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.transform(image)
        return image,


class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = Path(filename)
        self.columns = columns
        if self.filename.exists():
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, 'w')
            self.write(*self.columns)

    def write(self, *args):
        print(*args, sep=',', file=self.file, flush=True)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """A context manager that sets whether TF32 is allowed on cuDNN or matmul."""
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old


def get_safetensors_metadata(path):
    """Retrieves the metadata from a safetensors file."""
    return safetensors.safe_open(path, "pt").metadata()


def ema_update_dict(values, updates, decay):
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values
