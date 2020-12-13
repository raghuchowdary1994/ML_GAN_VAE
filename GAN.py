# -*- coding: utf-8 -*-


import torch
import torch.nn
import torch.nn.functional 
import torch.optim 
import torchvision 
import torch.autograd  
import torchvision.utils  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Generator(torch.nn.Module):
    def __init__(self, gan_input_dimension, gan_output_dimension):
        super(Generator, self).__init__()       
        self.fc1 = torch.nn.Linear(gan_input_dimension, 256)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, gan_output_dimension)
    
    # forward method
    def forward(self, x): 
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(d_input_dim, 1024)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc2(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.leaky_relu(self.fc3(x), 0.2)
        x = torch.nn.functional.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# building GAN
z_dimension = 100
mnist_dimension = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

GAN_generator = Generator(gan_input_dimension = z_dimension, gan_output_dimension = mnist_dimension).to(device)
GAN_discriminator = Discriminator(mnist_dimension).to(device)


# loss function
criterion = torch.nn.BCELoss() 

# network optimizer
lr = 0.0002 
Gen_optimizer = torch.optim.Adam(GAN_generator.parameters(), lr = lr)
Dis_optimizer = torch.optim.Adam(GAN_discriminator.parameters(), lr = lr)

# training the discriminiator 
def Dis_train(x):
    GAN_discriminator.zero_grad()
    
    x_real, y_real = x.view(-1, mnist_dimension), torch.ones(batch_size, 1)
    x_real, y_real = torch.autograd.Variable(x_real.to(device)), torch.autograd.Variable(y_real.to(device))

    Dis_output = GAN_discriminator(x_real)
    Dis_real_loss = criterion(Dis_output, y_real)
    Dis_real_score = Dis_output

    z = torch.autograd.Variable(torch.randn(batch_size, z_dimension).to(device))
    x_fake, y_fake = GAN_generator(z), torch.autograd.Variable(torch.zeros(batch_size, 1).to(device))

    Dis_output = GAN_discriminator(x_fake)
    Dis_fake_loss = criterion(Dis_output, y_fake)
    Dis_fake_score = Dis_output

    Dis_loss = Dis_real_loss + Dis_fake_loss
    Dis_loss.backward()
    Dis_optimizer.step()
        
    return  Dis_loss.data.item()

# training the generator 
def Gen_train(x):

    GAN_generator.zero_grad()

    z = torch.autograd.Variable(torch.randn(batch_size, z_dimension).to(device))
    y = torch.autograd.Variable(torch.ones(batch_size, 1).to(device))

    Gen_output = GAN_generator(z)
    Dis_output = GAN_discriminator(Gen_output)
    Gen_loss = criterion(Dis_output, y)

    Gen_loss.backward()
    Gen_optimizer.step()
        
    return Gen_loss.data.item()

# training the GAN for 300 epochs
n_epoch = 300
for epoch in range(1, n_epoch+1):           
    Dis_losses, Gen_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        Dis_losses.append(Dis_train(x))
        Gen_losses.append(Gen_train(x))
    
    if epoch % 20==0:
        with torch.no_grad():
            test_z = torch.autograd.Variable(torch.randn(64, z_dimension).to(device))
            generated = GAN_generator(test_z)
            torchvision.utils.save_image(generated.view(generated.size(0), 1, 28, 28), './gan-' + str(epoch) + '.png')

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(Dis_losses)), torch.mean(torch.FloatTensor(Gen_losses))))

with torch.no_grad():
    test_z = torch.autograd.Variable(torch.randn(64, z_dimension).to(device))
    generated = GAN_generator(test_z)
    torchvision.utils.save_image(generated.view(generated.size(0), 1, 28, 28), './gan' + '.png')