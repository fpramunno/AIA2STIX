# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:28:21 2023

@author: pio-r
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Define the encoder-decoder model
class AIA_EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: (1, 256, 256) -> (24, 2)
        self.encoder = nn.Sequential(
            # First conv block: 256x256 -> 128x128
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second conv block: 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third conv block: 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth conv block: 32x32 -> 16x16
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Fifth conv block: 16x16 -> 8x8
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # Flatten and project to latent space
            nn.AdaptiveAvgPool2d(1),  # 1024x1x1
            nn.Flatten(),  # 1024
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 24 * 2)  # 48 -> reshape to (24, 2)
        )
        
        # Decoder: (24, 2) -> (1, 256, 256)
        self.decoder = nn.Sequential(
            # Project from latent space
            nn.Linear(24 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        
        # Reshape and upsampling layers
        self.decoder_conv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def encode(self, x):
        """Encode input image to latent space"""
        latent_flat = self.encoder(x)
        return latent_flat.view(-1, 24, 2)
        
    def decode(self, z):
        """Decode latent space back to image"""
        z_flat = z.view(-1, 24 * 2)
        features = self.decoder(z_flat)
        features = features.view(-1, 1024, 8, 8)
        return self.decoder_conv(features)
        
    def forward(self, x):
        """Full forward pass: encode -> decode"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear( # linear projection to bring the time embedding to the proper dimension
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # projection
        return x + emb


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class PaletteModelV2(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None):
        super(PaletteModelV2, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        # self.sa1 = SelfAttention(self.image_size*2,int( self.true_img_size/2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.image_size*2, self.image_size*4)
        # self.sa2 = SelfAttention(self.image_size*4, int(self.true_img_size/4))
        self.down3 = Down(self.image_size*4, self.image_size*4)
        # self.sa3 = SelfAttention(self.image_size*4, int(self.true_img_size/8))
        
        # Bootleneck
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        self.bot3 = DoubleConv(self.image_size*8, 48)
        self.bot4 = DoubleConv(48, self.image_size*4)
        
        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        # self.sa4 = SelfAttention(self.image_size*2, int(self.true_img_size/4))
        self.up2 = Up(self.image_size*4, self.image_size)
        # self.sa5 = SelfAttention(self.image_size, int(self.true_img_size/2))
        self.up3 = Up(self.image_size*2, self.image_size)
        # self.sa6 = SelfAttention(self.image_size, self.true_img_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)    
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 

    def forward(self, x, y,  t, lab=None):
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) # Encoding timesteps is HERE, we provide the dimension we want to encode

        
        if lab is not None:
            t += self.label_emb(lab)
    
        # Concatenate the source image and reference image
        if y != None:
            x = torch.cat([x, y], dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x4 = self.bot4(x4)
        
        x = self.up1(x4, x3, t) # We note that upsampling box that in the skip connections from encoder 
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)

        return output
    

class VisibilityRefiner(nn.Module):
    def __init__(self, input_dim=48, hidden_dims=[128, 64], output_dim=48, dropout_rate=0.1, use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        layers = []
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: (batch_size, visibility_dim) where visibility_dim = 24*2 = 48
        refined = self.network(x)
        
        if self.use_residual:
            # Add residual connection
            return x + refined
        else:
            return refined

class ChiSquareLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted, target):
        # Chi-square distance: sum((predicted - target)^2 / (target + epsilon))
        # epsilon prevents division by zero for small values
        chi_square = torch.sum((predicted - target) ** 2 / (torch.abs(target) + self.epsilon), dim=-1)
        return torch.mean(chi_square)

class CompositeLoss(nn.Module):
    def __init__(self, primary_loss_type='mse', primary_weight=1.0, real_imag_weight=0.1, epsilon=1e-8):
        """
        Composite loss combining primary loss (MSE or chi-square) with separate MSE for real/imaginary parts.
        
        Args:
            primary_loss_type: 'mse' or 'chi_square' for the primary loss
            primary_weight: Weight for the primary loss function
            real_imag_weight: Weight for the real/imaginary MSE loss
            epsilon: Small value to prevent division by zero in chi-square loss
        """
        super().__init__()
        self.primary_weight = primary_weight
        self.real_imag_weight = real_imag_weight
        
        if primary_loss_type == 'mse':
            self.primary_loss = nn.MSELoss()
        elif primary_loss_type == 'chi_square':
            self.primary_loss = ChiSquareLoss(epsilon=epsilon)
        else:
            raise ValueError(f"Unsupported primary loss type: {primary_loss_type}")
        
        self.real_imag_mse = nn.MSELoss()
    
    def forward(self, predicted, target):
        # predicted and target shape: (batch_size, 48) where 48 = 24 * 2
        
        # Reshape to separate real and imaginary parts: (batch_size, 24, 2)
        predicted_reshaped = predicted.view(-1, 24, 2)
        target_reshaped = target.view(-1, 24, 2)
        
        # MSE loss on real parts (index 0)
        real_loss = self.real_imag_mse(predicted_reshaped[:, :, 0], target_reshaped[:, :, 0])
        
        # MSE loss on imaginary parts (index 1)
        imag_loss = self.real_imag_mse(predicted_reshaped[:, :, 1], target_reshaped[:, :, 1])
        
        # Combine real and imaginary MSE losses
        real_imag_loss = (self.primary_weight * real_loss + self.real_imag_weight * imag_loss) / 2
        
        return real_imag_loss

class PaletteModelV2_WithLatentBottleneck(nn.Module):
    def __init__(self, c_in=1, c_out=1, image_size=64, time_dim=256, device='cuda', latent=False, true_img_size=64, num_classes=None):
        super(PaletteModelV2_WithLatentBottleneck, self).__init__()

        # Encoder
        self.true_img_size = true_img_size
        self.image_size = image_size
        self.time_dim = time_dim
        self.device = device
        self.inc = DoubleConv(c_in, self.image_size) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.image_size, self.image_size*2) # input and output channels
        self.down2 = Down(self.image_size*2, self.image_size*4)
        self.down3 = Down(self.image_size*4, self.image_size*4)
        
        # Original bottleneck layers
        self.bot1 = DoubleConv(self.image_size*4, self.image_size*8)
        self.bot2 = DoubleConv(self.image_size*8, self.image_size*8)
        
        # NEW: Latent space bottleneck (24, 2)
        # Calculate the spatial dimensions after downsampling
        # Assuming input is 256x256 and we have 3 down layers (each /2): 256 -> 128 -> 64 -> 32
        spatial_dim_after_down = true_img_size // (2**3)  # 32 for 256x256 input
        bottleneck_features = self.image_size * 8 * spatial_dim_after_down * spatial_dim_after_down
        
        # Project to latent space (24, 2) = 48 dimensions
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling -> (batch, channels, 1, 1)
            nn.Flatten(),  # -> (batch, channels)
            nn.Linear(self.image_size*8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 48),  # 24 * 2 = 48
        )
        
        # Project back from latent space to feature maps
        self.from_latent = nn.Sequential(
            nn.Linear(48, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.image_size*8 * spatial_dim_after_down * spatial_dim_after_down),
            nn.ReLU(inplace=True),
        )
        
        self.spatial_dim_after_down = spatial_dim_after_down
        
        # Continue with original bottleneck
        self.bot3 = DoubleConv(self.image_size*8, self.image_size*4)
        
        # Decoder: reverse of encoder
        self.up1 = Up(self.image_size*8, self.image_size*2)
        self.up2 = Up(self.image_size*4, self.image_size)
        self.up3 = Up(self.image_size*2, self.image_size)
        self.outc = nn.Conv2d(self.image_size, c_out, kernel_size=1) # projecting back to the output channel dimensions
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        if latent == True:
            self.latent = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256)).to(device)    
  
    def pos_encoding(self, t, channels):
        """
        Input noised images and the timesteps. The timesteps will only be
        a tensor with the integer timesteps values in it
        """
        inv_freq = 1.0 /  (
            10000 
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc 

    def encode_to_latent(self, x, y, lab, t):
        """Encode input to latent space (24, 2)"""
        # Pass the source image through the encoder network
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        if lab is not None:
            t += self.label_emb(lab)
        
        # Concatenate the source image and reference image
        if y != None:  
            x = torch.cat([x, y], dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        
        # NEW: Project to latent space (24, 2)
        latent_flat = self.to_latent(x4)  # (batch, 48)
        latent = latent_flat.view(-1, 24, 2)  # (batch, 24, 2)
        
        return latent, (x1, x2, x3, x4)  # Return latent and skip connections
    
    def decode_from_latent(self, latent, skip_connections, lab, t):
        """Decode from latent space (24, 2) back to image"""
        x1, x2, x3, x4_orig = skip_connections
        
        # Flatten latent and project back to feature maps
        latent_flat = latent.view(-1, 48)  # (batch, 48)
        x4_reconstructed = self.from_latent(latent_flat)  # (batch, features)
        x4_reconstructed = x4_reconstructed.view(-1, self.image_size*8, 
                                               self.spatial_dim_after_down, 
                                               self.spatial_dim_after_down)
        
        x4 = self.bot3(x4_reconstructed)
        
        # Decoder with skip connections
        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        
        return output

    def forward(self, x, y=None, lab=None, t=None, return_latent=False):
        """
        Forward pass that can return both reconstruction and latent space
        
        Args:
            x: Input image
            y: Reference image (can be None for autoencoder mode)
            lab: Labels (can be None)
            t: Time steps (can be None, will use zeros)
            return_latent: If True, returns (output, latent), else just output
        """
        # Handle default values for autoencoder mode
        batch_size = x.shape[0]
        
        if y is None:
            y = None  # Use input as reference for autoencoder
        if t is None:
            t = torch.zeros(batch_size, device=x.device)  # Default timestep
        
        # Encode to latent space
        latent, skip_connections = self.encode_to_latent(x, y, lab, t)
        
        # Decode back to image
        output = self.decode_from_latent(latent, skip_connections, lab, t)
        
        if return_latent:
            return output, latent  # (batch, c_out, H, W), (batch, 24, 2)
        else:
            return output  # (batch, c_out, H, W)
