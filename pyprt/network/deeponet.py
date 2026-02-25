from ..needs import *
from .funcnet import FCN

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

    def forward(self,ltaun,tn,pgtopn):
        """
        ltau   : (Nt,) # (log(tau)+4)/(2+4)*2-1
        tn     : (Nb,Ns), Ns=64 # (T-1000)/(10000-1000)*2-1
        pgtopn : (Nb,) # (log(Pg_top)-2)/(7-2)*2-1
        """
        f1 = self.branch_net1(tn.unsqueeze(1)) # (Nb,1,128)
        f2 = self.branch_net2(pgtopn.unsqueeze(1)).unsqueeze(1) # (Nb,1,128)
        f3 = self.trunck_net(ltaun.unsqueeze(1)).unsqueeze(0) # (1,Nt,32)
        f12 = torch.cat([f1,f2],dim=1) # (Nb,2,128)
        f4 = self.agency_net(f12) # (Nb,1,32)
        f = torch.sum(f3*f4,dim=-1) # (Nb,Nt)
        f = f-f[:,0:1]+pgtopn.unsqueeze(1)
        return f

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