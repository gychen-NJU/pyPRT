from ..needs import *

class FixedFNOHydrostatic(nn.Module):
    """
    Fixed FNO for hydrostatic solver
    """
    
    def __init__(self, modes=16, width=128, depth=8, dropout=0.1, activation='gelu'):
        super().__init__()
        
        # 1. Enhanced input boost layer
        self.lift = nn.Sequential(
            nn.Linear(3, width * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width)
        )
        # 2. Learnable sine positional encoding (Fixed)
        self.pos_encoding = LearnablePositionalEncodingFixed(width, max_len=1000)
        # 3. Endanced FNO blocks
        self.fno_layers = nn.ModuleList([
            EnhancedFNOBlockFixed(width, modes, width, activation=activation, dropout=dropout)
            for _ in range(depth)
        ])
        # 4. Adaptive feature fusion layer
        self.feature_fusion = AdaptiveFeatureFusionFixed(width, depth)
        # 5. Fixed multi-scale projection head
        self.project = MultiScaleProjectionFixed(width, scale_factors=[1, 2, 4])
        # 6. Improved boundary condition handler
        self.boundary_net = nn.Sequential(
            nn.Linear(1, width * 2),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width)
        )
        # 7. Debugging marker
        self.debug_mode = False
        
    def _get_activation(self, activation):
        """Getting activation function"""
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'mish':
            return nn.Mish()
        else:
            return nn.GELU()
    
    def set_debug_mode(self, debug=True):
        """Setting debugging mode"""
        self.debug_mode = debug
        
    def forward(self, ltau, T, Pg_top):
        """
        ltau  : in [-1,1]: (log(tau)+4)/(2+4)*2-1
        T     : in [-1,1]: (T-1000)/(10000-1000)*2-1
        Pg_top: in [-1,1]: (log(Pg_top)-2)/(7-2)*2-1
        """
        Nb, Nt = T.shape  
        if self.debug_mode:
            print(f"Input: ltau={ltau.shape}, T={T.shape}, Pg_top={Pg_top.shape}")
        # reconstruct input tensor
        tau_grid = ltau.unsqueeze(0).expand(Nb, -1).unsqueeze(-1)  # (Nb, Nt, 1)
        T_input = T.unsqueeze(-1)  # (Nb, Nt, 1)
        # Create boundary condition mask (1 at top and 0 at rest)
        boundary_mask = torch.zeros(Nb, Nt, 1, device=T.device)
        boundary_mask[:, 0, :] = 1.0
        # Concatenate all features
        x = torch.cat([tau_grid, T_input, boundary_mask], dim=-1)  # (Nb, Nt, 3)
        if self.debug_mode:
            print(f"Concatenated input: {x.shape}")
        # FNO processing
        x = self.lift(x)
        if self.debug_mode:
            print(f"After lifting: {x.shape}")
        # Add positional encoding
        x = x + self.pos_encoding(x)
        if self.debug_mode:
            print(f"After positional encoding: {x.shape}")
        # store features for deep fusion
        layer_features = []
        # Transfer by FNO layers
        for i, layer in enumerate(self.fno_layers):
            x = layer(x)
            layer_features.append(x.clone())  # store features
            if self.debug_mode:
                print(f"After FNO layer {i+1}: {x.shape}")
        # Adaptive feature fusion
        x = self.feature_fusion(layer_features)
        if self.debug_mode:
            print(f"After feature fusion: {x.shape}")
        # Process boundary conditions
        boundary_feat = self.boundary_net(Pg_top.unsqueeze(-1))
        boundary_feat = boundary_feat.unsqueeze(1).expand(-1, Nt, -1)
        # Gate mechanism to fuse boundary information
        gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        x = x + gate * boundary_feat
        if self.debug_mode:
            print(f"After boundary fusion: {x.shape}")
        # Multi-scale projection
        Pg = self.project(x)
        if self.debug_mode:
            print(f"After projection: {Pg.shape}")
        # Force top boundary condition
        Pg = Pg - Pg[:, 0:1] + Pg_top.unsqueeze(1)
        if self.debug_mode:
            print(f"After boundary adjustment: {Pg.shape}")
        return Pg

class LearnablePositionalEncodingFixed(nn.Module):
    """
    Learnable sine positional encoding (Fixed)
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.encoding, mean=0, std=0.02)
        
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.encoding.size(1):
            # dynamically expand allocated length if required position exceeds the pre-allocated length
            new_encoding = torch.zeros(1, seq_len, self.encoding.size(2), 
                                     device=self.encoding.device)
            new_encoding[:, :self.encoding.size(1), :] = self.encoding
            self.encoding = nn.Parameter(new_encoding)
        return x + self.encoding[:, :seq_len, :]

class EnhancedFNOBlockFixed(nn.Module):
    """FNO Block (Enhanced, Fixed)"""
    
    def __init__(self, in_channels, modes, width, activation='gelu', dropout=0.1):
        super().__init__()
        self.modes = modes
        self.width = width    
        # fourier layer
        self.fourier = EnhancedSpectralConv1dFixed(width, width, modes, activation)   
        # local convolutional layer
        self.conv_local = nn.Sequential(
            nn.Conv1d(width, width * 2, 1),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(width * 2, width, 1)
        )
        # adaptive gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Sigmoid()
        )
        # feedforward network
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width)
        )
        # layer normalization
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        # residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def _get_activation(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x):
        # input: (B, N, C)
        B, N, C = x.shape
        residual = x
        # normalize input
        x_norm = self.norm1(x)
        # fourier branch
        x_ft = x_norm.permute(0, 2, 1)  # (B, C, N)
        x_ft = self.fourier(x_ft)
        x_ft = x_ft.permute(0, 2, 1)  # (B, N, C)
        # local branch
        x_local = self.conv_local(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        # adaptive gating fusion
        combined = torch.cat([x_ft, x_local], dim=-1)
        gate = self.gate(combined)
        x_fused = gate * x_ft + (1 - gate) * x_local
        # first residual connection
        x = residual + self.residual_scale * x_fused
        # second residual connection (feedforward network)
        residual2 = x
        x = self.norm2(x)
        x_ff = self.ff(x)
        x = residual2 + x_ff
        return x

class EnhancedSpectralConv1dFixed(nn.Module):
    """1D Spectral Convolution"""
    def __init__(self, in_channels, out_channels, modes, activation='silu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        # learnable spectral weights
        self.scale = 1.0 / (in_channels + out_channels)
        # real and imaginary parts of weights
        self.weight_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )
        # activation function
        self.activation = activation
        
    def forward(self, x):
        batchsize, in_channels, length = x.shape
        # fourier transform
        x_ft = torch.fft.rfft(x, dim=2)
        # only process low-order modes
        modes = min(self.modes, x_ft.shape[2])
        out_ft = torch.zeros(batchsize, self.out_channels, 
                            x_ft.shape[2], 
                            dtype=torch.complex64, 
                            device=x.device)
        for i in range(modes):
            out_ft[:, :, i] = torch.einsum(
                "bi,io->bo", 
                x_ft[:, :, i], 
                torch.complex(self.weight_real[:, :, i], self.weight_imag[:, :, i])
            )
        # apply activation function (in frequency domain)
        if self.activation == 'silu':
            out_ft = F.silu(out_ft.real) + 1j * F.silu(out_ft.imag)
        elif self.activation == 'gelu':
            out_ft = F.gelu(out_ft.real) + 1j * F.gelu(out_ft.imag)   
        # inverse fourier transform
        out = torch.fft.irfft(out_ft, n=length, dim=2)
        return out

class AdaptiveFeatureFusionFixed(nn.Module):
    """Adaptive feature fusion module"""
    def __init__(self, width, depth):
        super().__init__()
        self.width = width
        self.depth = depth
        # learning weights for each layers
        self.layer_weights = nn.Parameter(torch.ones(depth))
        self.layer_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.LayerNorm(width)
            ) for _ in range(depth)
        ])
        # fusion transform
        self.fusion_transform = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width)
        )
    
    def forward(self, layer_features):
        # calculate layer weights
        weights = torch.softmax(self.layer_weights, dim=0)
        # weighted fusion
        fused = 0
        for i, feat in enumerate(layer_features):
            transformed = self.layer_transform[i](feat)
            fused = fused + weights[i] * transformed
        # final transform
        output = self.fusion_transform(fused)
        return output

class MultiScaleProjectionFixed(nn.Module):
    """Multi scale projection header"""
    def __init__(self, width, scale_factors=[1, 2, 4]):
        super().__init__()
        self.scale_factors = scale_factors
        # multi scale convolutional layer
        self.convs = nn.ModuleList()
        for scale in scale_factors:
            kernel_size = scale * 2 + 1
            conv = nn.Sequential(
                nn.Conv1d(width, width, kernel_size, padding=scale),
                nn.GELU(),
                nn.Conv1d(width, width // len(scale_factors), 1)
            )
            self.convs.append(conv)
        # calculate total channels after concatenation
        total_channels = (width // len(scale_factors)) * len(scale_factors) 
        # dynamic fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, 1)
        )
        # debug information
        print(f"MultiScaleProjection: width={width}, total_channels={total_channels}")
    
    def forward(self, x):
        # x: (B, N, width)
        B, N, C = x.shape
        # transpose to perform convolution
        x_t = x.permute(0, 2, 1)  # (B, C, N)
        # multi scale feature extraction
        features = []
        for conv in self.convs:
            feat = conv(x_t)  # (B, C_out, N)
            feat = feat.permute(0, 2, 1)  # (B, N, C_out)
            features.append(feat)
        # concatenate multi scale features
        combined = torch.cat(features, dim=-1)  # (B, N, total_channels)
        # final projection
        output = self.fusion(combined).squeeze(-1)  # (B, N)
        return output