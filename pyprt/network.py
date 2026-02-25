from .needs import *
from .math  import *

class FCN(nn.Module):
    def __init__(self, layers):
        super(FCN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = nn.Tanh()

        layer_list = list()
        layer_list.append(
            ('layer_%d' % 0, torch.nn.Linear(layers[0], layers[0+1]))
        )
        n0 = layer_list[0][1]
        torch.nn.init.xavier_uniform_(n0.weight)

        layer_list.append(('activation_%d' % 0, self.activation))

        for i in range(1, self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        nL = layer_list[-1][1]
        torch.nn.init.xavier_uniform_(nL.weight)
#         layer_list.append('activateion_%d' % (self.depth - 1), nn.Sigmoid())

        layerDict = OrderedDict(layer_list)
        self.layers = nn.Sequential(layerDict)

    def forward(self, x):
#         out = nn.Softmax(dim=1)(self.layers(x))
        out = (self.layers(x))
        return out

class FuncNet(nn.Module):
    def __init__(self, hidden_layers=[16,32,64,64,32,16]):
        super().__init__()
        self.net = FCN([2]+hidden_layers+[1])

    def forward(self, u, a):
        inputs = torch.stack([u,torch.log10(a)],dim=-1)
        return self.net(inputs).squeeze(-1)

class Training_history():
    def __init__(self, loss_list,optimizer,scheduler):
        self.loss_list = loss_list
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__():
        return self.loss_list

    def save(self, path, model=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_data = dict(
            loss_list = self.loss_list,
            optimizer_state = self.optimizer.state_dict() if self.optimizer else None,
            scheduler_state = self.scheduler.state_dict() if self.scheduler else None,
            model_state = model.state_dict() if model else None,
        )
        with open(path,'wb') as f:
            pickle.dump(save_data, f)
        print(f"Save training history to {path}")

    @classmethod
    def load(cls, path, model=None, device=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training history file {path} is not found")
        with open(path, 'rb') as f:
            load_data = pickle.load(f)

        loss_list = load_data['loss_list']
        optimizer_state = load_data['optimizer_state']
        scheduler_state = load_data['scheduler_state']
        model_state = load_data['model_state']

        model = FuncNet() if model is None else model
        if model_state:
            model.load_state_dict(model_state)
        
        if optimizer_state:
            optimizer = optim.AdamW(model.parameters())
            optimizer.load_state_dict(optimizer_state)
        else:
            optimizer = None

        if scheduler_state and optimizer:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
            scheduler.load_state_dict(scheduler_state)
        else:
            scheduler = None
            
        history = cls(loss_list, optimizer, scheduler)
        return history
            

def train(
    net,
    dataset,
    epoches=1000,
    batch_size=10000,
    **kwargs
):
    print_interval = kwargs.get('print_interval', 100)
    save_interval  = kwargs.get('save_interval', 10000)
    save_name      = kwargs.get('save_name', 'network')
    check_batch    = kwargs.get('check_batch', False)
    directory      = os.path.dirname(save_name)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"!Create directory: {directory}")
    device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr = kwargs.get('lr',1e-3),
        betas = kwargs.get('betas', (0.9,0.999)),
        eps = kwargs.get('eps', 1e-8),
        weight_decay = kwargs.get('weight_decay', 0.01)
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=kwargs.get('step_size', 100),
        gamma=kwargs.get('gamma', 0.5)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    time_start = time.time()

    loss_list = []
    for iepoch in range(1, epoches+1):
        net.train()
        loss = []
        for ibatch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predct = net(inputs[:,0],inputs[:,1])
            iloss  = criterion(predct, labels)
            if check_batch and (iepoch==1):
                print(f"Batch: {ibatch:5d} | Loss: {iloss.item():.5e}")
                if torch.any(torch.isnan(iloss)):
                    print(f"inputs NaN: {torch.any(torch.isnan(inputs))}")
                    print(f"labels NaN: {torch.any(torch.isnan(labels))}")
                    print(f"predct NaN: {torch.any(torch.isnan(predct))}")
                    raise ValueError("NaN values found in the forward pass")
            loss.append(iloss.item())
            optimizer.zero_grad()
            iloss.backward()
            optimizer.step()
        loss_list.append(np.mean(loss))
        if iepoch==1 or iepoch==epoches or (iepoch%print_interval==0):
            time_now = time.time()
            print(f"Epoch: {iepoch:5d} | Loss: {loss_list[-1]:.5e} | LR: {scheduler.get_last_lr()[0]:.5e} | "
                  f"Time: {(time_now-time_start)/60:7.3f} [min]")
        if iepoch==epoches or (iepoch%save_interval==0):
            net.train()
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(labels)
                    _ = net(inputs[:,0], inputs[:,1])
            net.eval()
            torch.save(net,save_name+f"_{iepoch:04d}.pkl")
        scheduler.step()

    net.train()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(labels)
            _ = net(inputs[:,0], inputs[:,1])
    net.eval()

    print('!! Finish Training !!')
    history = Training_history(loss_list, optimizer, scheduler)
    return history

class UserDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels=labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

class FixedFNOHydrostatic(nn.Module):
    """
    Fixed FNO for hydrostatic solver
    """
    
    def __init__(self, modes=16, width=128, depth=6, dropout=0.1, activation='gelu'):
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

class HSE_DeepONet(nn.Module):
    """ DeepONet for HSE solver """
    def __init__(self):
        super().__init__()
        self.branch_net1 = Conv1dSEAttnDecoder_TransposeConv(
            in_channels=1, mid_channels=64, out_channels=1, 
            kernel_size=3, layers=4, se_reduction=16
            ) # (Nb,1,64) -> (Nb,1,128) : decoding temperature
        self.branch_net2 = FCN([1,16]+[32]*2+[64]*2+[128]) # (Nb,1)->(Nb,128) : process boundary
        self.trunck_net = FCN([1]+[16]*4+[32]) # (Nb,1) -> (Nb,32) : decoding log(tau)
        self.agency_net = Conv1dSEAttnDecoder_Downsample(
            in_channels=2, mid_channels=64, out_channels=1, 
            kernel_size=3, layers=4, se_reduction=16
            ) # (Nb,2,128) -> (Nb,1,32) : decoding incorporated feature from Temp and B.C.

class Conv1dSEAttnDecoder_Downsample(nn.Module):
    """
    1D conv + SE attentional decoder with downsampling
    inputs: (Nb, 2, 128) -> outputs: (Nb, 1, 32)
    """
    def __init__(self, in_channels=2, mid_channels=64, out_channels=1, 
                 kernel_size=3, layers=4, se_reduction=16, downsample_factor=4):
        super().__init__()
        self.layers = layers
        self.downsample_factor = downsample_factor  # downsampling factor: 4 (128→32)
        # 1st layer: input channel:2 -> mid_channels，using stride=2 for downsampling (128→64)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size, 
                              stride=2, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        # 2nd layer: continue downsampling (64→32)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size, 
                              stride=2, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        # subsequent layers: keep length 32
        self.convs = nn.ModuleList([
            nn.Conv1d(mid_channels, mid_channels, kernel_size, 
                     padding=kernel_size//2)
            for i in range(layers-2)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(mid_channels) for _ in range(layers-2)])
        # SE attentional modules
        self.se_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(mid_channels, mid_channels // se_reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels // se_reduction, mid_channels, 1),
                nn.Sigmoid()
            ) for _ in range(layers)
        ])
        # final 1x1 convolution: mid_channels -> out_channels
        self.final_conv = nn.Conv1d(mid_channels, out_channels, 1)
        
    def forward(self, x):
        # x: (Nb, 2, 128)
        # 1st layer: downsampling 128→64
        x = F.relu(self.bn1(self.conv1(x)))  # (Nb, mid_channels, 64)
        se_weight = self.se_modules[0](x)
        x = x * se_weight
        # 2nd layer: downsampling 64→32
        x = F.relu(self.bn2(self.conv2(x)))  # (Nb, mid_channels, 32)
        se_weight = self.se_modules[1](x)
        x = x * se_weight
        # subsequent layers: keep length 32
        for i in range(self.layers-2):
            conv_out = self.convs[i](x)
            conv_out = self.bns[i](conv_out)
            conv_out = F.relu(conv_out)
            
            se_weight = self.se_modules[i+2](conv_out)
            conv_out = conv_out * se_weight
            
            x = conv_out
        # final 1x1 convolution: mid_channels -> out_channels
        x = self.final_conv(x)  # (Nb, 1, 32)
        return x

class Conv1dSEAttnDecoder_TransposeConv(nn.Module):
    """
    Upsampling with Transpose Conv
    """
    def __init__(self, in_channels=1, mid_channels=64, out_channels=1, 
                 kernel_size=3, layers=4, se_reduction=16):
        super().__init__()
        self.layers = layers
        # The first half: feature extraction
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels if i == 0 else mid_channels,
                      mid_channels,
                      kernel_size,
                      padding=kernel_size//2)
            for i in range(layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(mid_channels) for _ in range(layers)])
        # SE attentional modules
        self.se_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(mid_channels, mid_channels // se_reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels // se_reduction, mid_channels, 1),
                nn.Sigmoid()
            ) for _ in range(layers)
        ])
        # The second half: upsampling with transpose conv
        self.upsample = nn.ConvTranspose1d(
            mid_channels, mid_channels, 
            kernel_size=4, stride=2, padding=1
        )
        self.upsample_bn = nn.BatchNorm1d(mid_channels)
        # The third half: post-upsampling convolution
        self.post_upsample_conv = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.post_upsample_bn = nn.BatchNorm1d(mid_channels)
        # final 1x1 convolution: mid_channels -> out_channels
        self.final_conv = nn.Conv1d(mid_channels, out_channels, 1)
        
    def forward(self, x):
        # x: (Nb, 1, 64)
        for i in range(self.layers):
            # conv + BN + ReLU
            conv_out = self.convs[i](x)  # (Nb, mid_channels, 64)
            conv_out = self.bns[i](conv_out)
            conv_out = F.relu(conv_out)
            # SE attention
            se_weight = self.se_modules[i](conv_out)  # (Nb, mid_channels, 1)
            conv_out = conv_out * se_weight
            x = conv_out
        # The third half: upsampling with transpose conv: 64 → 128
        x = F.relu(self.upsample_bn(self.upsample(x)))  # (Nb, mid_channels, 128)
        # The fourth half: post-upsampling convolution
        x = F.relu(self.post_upsample_bn(self.post_upsample_conv(x)))
        # final 1x1 convolution: mid_channels -> out_channels
        x = self.final_conv(x)  # (Nb, 1, 128)
        return x

    def forward(self,ltaun,tn,pgtopn):
        """
        ltau   : (Nt,)
        tn     : (Nb,Ns), Ns=64
        pgtopn : (Nb,)
        """
        f1 = self.branch_net1(tn.unsqueeze(1)) # (Nb,1,128)
        f2 = self.branch_net2(pgtopn.unsqueeze(1)).unsqueeze(1) # (Nb,1,128)
        f3 = self.trunck_net(ltaun.unsqueeze(1)).unsqueeze(0) # (1,Nt,32)
        f12 = torch.cat([f1,f2],dim=1) # (Nb,2,128)
        f4 = self.agency_net(f12) # (Nb,1,32)
        f = torch.sum(f3*f4,dim=-1) # (Nb,Nt)
        f = f-f[:,0:1]+pgtopn.unsqueeze(1)
        return f

if __name__ == '__main__':
    uu       = torch.linspace(-10,10,1000)
    aa       = torch.pow(10,torch.linspace(-5,1.5,1000))
    U,A      = torch.meshgrid(uu,aa,indexing='ij')
    u_sample = U.flatten()
    a_sample = A.flatten()
    H_sample = torch.log(VoigtFunction(u_sample[None,:],a_sample[None,:],ynodes=1000)[0])
    F_sample = VoigtFaradayFunction(u_sample[None,:],a_sample[None,:],ynodes=1000)[0]

    ua_data = torch.from_numpy(np.stack([u_sample,a_sample],axis=-1))
    Voigt_dataset = UserDataset(ua_data, H_sample)
    VoigtFaraday_dataset = UserDataset(ua_data, F_sample)

    VoigtNet = FuncNet()
    history=train(
        VoigtNet,
        Voigt_dataset,
        epoches=1000,
        batch_size = 1000,
        save_name = './VoigtNet/voigt',
        gamma=0.95,
        step_size=10,
        lr=1e-2,
        device='cuda:0',
        print_interval=50,
        betas=(0.5,0.99)
    )
    history.save('./VoigtNet/history.pkl')

    VoigtFaradayNet = FuncNet()
    history=train(
        VoigtFaradayNet,
        VoigtFaraday_dataset,
        epoches=1000,
        batch_size = 1000,
        save_name = './VoigtFaradayNet/voigtfaraday',
        gamma=0.95,
        step_size=10,
        lr=1e-2,
        device='cuda:0',
        print_interval=50,
        betas=(0.5,0.99)
    )
    history.save('./VoigtFaradayNet/history.pkl')