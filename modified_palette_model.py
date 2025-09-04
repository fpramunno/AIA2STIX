import torch
import torch.nn as nn

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
            y = x  # Use input as reference for autoencoder
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


# You'll also need to define these helper classes if they're not already defined:
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x, t=None):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, t=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)