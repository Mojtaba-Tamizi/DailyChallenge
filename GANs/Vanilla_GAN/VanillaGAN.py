import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


class Dicriminator(nn.Module):
    def __init__(self, img_dim) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim = 64
img_dim = 28 * 28
batch_size = 32
num_epochs = 50

disc = Dicriminator(img_dim=img_dim).to(device=device)
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device=device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device=device)
transforms = T.Compose(
    [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]
)
dataset = datasets.MNIST(root='../../dataset', transform=transforms, download=True, train=True)
dataLoader = DataLoader(dataset, batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataLoader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_real))
        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}] \ '
                f'Loss D: {lossD:.4f}, Loss G: {lossG:.4f}'
            )

            with torch.no_grad():
                fake = gen(fixed_noise).view(-1, 1, 28, 28)
                data = real.view(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real= torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    'Mnist Fake Images', img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    'Mnist Real Images', img_grid_real, global_step=step
                )

                step += 1