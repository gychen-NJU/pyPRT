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