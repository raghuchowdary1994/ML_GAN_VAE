# -*- coding: utf-8 -*-

import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torchvision
import torchvision.utils 

batch_size = 100

train_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=False, transform=torchvision.transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# building VAE
class VAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        # endcoder
        self.fc1 = torch.nn.Linear(x_dim, h_dim1)
        self.fc2 = torch.nn.Linear(h_dim1, h_dim2)
        self.fc31 = torch.nn.Linear(h_dim2, z_dim)
        self.fc32 = torch.nn.Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = torch.nn.Linear(z_dim, h_dim2)
        self.fc5 = torch.nn.Linear(h_dim2, h_dim1)
        self.fc6 = torch.nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x):
        h = torch.nn.functional.relu(self.fc1(x))
        h = torch.nn.functional.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.rand_like(std)
        return eps.mul(std).add(mu)
        
    def decoder(self, z):
        h = torch.nn.functional.relu(self.fc4(z))
        h = torch.nn.functional.relu(self.fc5(h))
        return torch.nn.functional.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 28*28))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
vae.cuda()

optimizer = torch.optim.Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var- mu.pow(2)- log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print(' Test set loss: {:.4f}'.format(test_loss))

# training the VAE for 300 epochs
for epoch in range(1, 300):
    train(epoch)
    test()
    if epoch % 20==0:
        with torch.no_grad():
            z = torch.randn(64, 2).cuda()
            sample = vae.decoder(z).cuda()
            torchvision.utils.save_image(sample.view(64, 1, 28, 28), './vae-'+ str(epoch) + '.png')

with torch.no_grad():
            z = torch.randn(64, 2).cuda()
            sample = vae.decoder(z).cuda()
            torchvision.utils.save_image(sample.view(64, 1, 28, 28), './vae-'+  '.png')