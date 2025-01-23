import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from utils import DEVICE, LOG_DIR, initialize_weights, gradient_penalty, clear_directories, check_gradients, check_dead_neurons, plot_losses, save_fake_images, create_gif, save_checkpoint


# Hyperparams
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
NOISE_DIM = 100
CRITIC_FEATURES = 16
GENERATOR_FEATURES = 16
BATCH_SIZE = 64
EPOCH_NUM = 10
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Here are different sets of learning rates for you to experiment with
# Set 1: Weaker Balanced Training
CRITIC_LEARNING_RATE = 1e-4
GENERATOR_LEARNING_RATE = 1e-4

# Set 2: Stronger Balanced Training
# CRITIC_LEARNING_RATE = 2e-4
# GENERATOR_LEARNING_RATE = 2e-4

# Set 3: Weaker Critic, Stronger Generator
# CRITIC_LEARNING_RATE = 1e-4
# GENERATOR_LEARNING_RATE = 5e-4

# Set 4: Stronger Critic, Weaker Generator
# CRITIC_LEARNING_RATE = 5e-4
# GENERATOR_LEARNING_RATE = 1e-4

# Set 5: More Critic Updates (Weaker Generator)
# CRITIC_LEARNING_RATE = 3e-4
# GENERATOR_LEARNING_RATE = 2e-4

# Set 6: Less Critic Updates (Weaker Critic)
# CRITIC_LEARNING_RATE = 1e-4
# GENERATOR_LEARNING_RATE = 3e-4

# Set 7: Aggressive Training for Both
# CRITIC_LEARNING_RATE = 5e-4
# GENERATOR_LEARNING_RATE = 5e-4


class Discriminator(nn.Module):
    def __init__(self, img_channels, disc_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, disc_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            self._block(disc_features, disc_features * 2, 4, 2, 1),
            self._block(disc_features * 2, disc_features * 4, 4, 2, 1),
            self._block(disc_features * 4, disc_features * 8, 4, 2, 1),

            nn.Conv2d(disc_features * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, img_channels, gen_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise, gen_features * 16, 4, 1, 0),
            self._block(gen_features * 16, gen_features * 8, 4, 2, 1),
            self._block(gen_features * 8, gen_features * 4, 4, 2, 1),
            self._block(gen_features * 4, gen_features * 2, 4, 2, 1),

            nn.ConvTranspose2d(gen_features * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.gen(x)


def train_dcgan(discriminator, generator, fixed_noise, loader, disc_opt, gen_opt, criterion, writer_fake, writer_real, writer_loss):
    step = 0
    disc_losses, gen_losses = [], []
    
    discriminator.train()
    generator.train()
    
    for epoch in range(EPOCH_NUM):
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        
        for batch_idx, (real, _) in enumerate(loader):

            """
            Discriminator Training

            The discriminator learns by:
            1. Getting real images -> predict 1
            2. Getting fake images -> predict 0
            3. Update weights to improve accuracy
            """
            
            real = real.to(DEVICE)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)
            fake = generator(noise)  # Create fake images
            
            disc_real = discriminator(real).reshape(-1)  # Discriminator prediction for real images
            disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))  # Compare with 1 (real)

            disc_fake = discriminator(fake.detach()).reshape(-1)  # Discriminator prediction for fake images
            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))  # Compare with 0 (fake)

            disc_loss = (disc_real_loss + disc_fake_loss) / 2  # Average loss for stability
            epoch_disc_loss += disc_loss.item()
            # disc_losses.append(disc_loss.item())
            writer_loss.add_scalar("Discriminator Loss", disc_loss.item(), global_step=epoch * len(loader) + batch_idx)

            discriminator.zero_grad()  # Reset gradients
            disc_loss.backward()  # Compute gradients
            disc_opt.step()  # Update discriminator weights
            
            
            """
            Generator Training

            The generator learns by:
            1. Trying to fool the discriminator to predict 1 for fake images.
            """

            output = discriminator(fake).reshape(-1)  # Get discriminator output (0 or 1)
            gen_loss = criterion(output, torch.ones_like(output))  # Fool discriminator into predicitng 1
            epoch_gen_loss += gen_loss.item()
            # gen_losses.append(gen_loss.item())
            writer_loss.add_scalar("Generator Loss", gen_loss.item(), global_step=epoch * len(loader) + batch_idx)

            generator.zero_grad()  # Reset gradients
            gen_loss.backward()  # Compute gradients
            gen_opt.step()  # Update generator weights
            
            if batch_idx % 100 == 0:
                print(f"EPOCH[{epoch}/{EPOCH_NUM}], "
                    f"BATCH[{batch_idx}/{len(loader)}]"
                    f"DISC LOSS: {disc_loss:.2f}, "
                    f"GEN LOSS: {gen_loss:.2f}"
                )
                with torch.no_grad():
                    fake = generator(fixed_noise)

                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                    writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                    writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)

                step += 1
                
        avg_disc_loss = epoch_disc_loss / len(loader)
        avg_gen_loss = epoch_gen_loss / len(loader)
        disc_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)
                    
        if epoch % 5 == 0:
            save_checkpoint(epoch, generator, discriminator, gen_opt, disc_opt)
            check_gradients(generator)
            check_gradients(discriminator)
                    
        save_fake_images(generator, fixed_noise, epoch)
                    
    plot_losses(disc_losses, gen_losses)
    
    
def train_wgan(critic, generator, fixed_noise, loader, opt_critic, opt_gen, writer_fake, writer_real, writer_loss):
    step = 0
    critic_losses, gen_losses = [], []
    
    critic.train()
    generator.train()
    
    for epoch in range(EPOCH_NUM):
        critic_loss_epoch = 0.0
        gen_loss_epoch = 0.0
        
        for batch_idx, (real, _) in enumerate(tqdm(loader)):

            """
            Critic Training

            The critic learns by:
            1. Getting real images -> predict 1
            2. Getting fake images -> predict 0
            3. Update weights to improve accuracy
            """
            
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]
            
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, NOISE_DIM, 1, 1).to(DEVICE)
                fake = generator(noise)  # Create fake images
            
                critic_real = critic(real).reshape(-1)  # Critic prediction for real images
                critic_fake = critic(fake).reshape(-1)  # Critic prediction for fake images

                gp = gradient_penalty(critic, real, fake)

                critic_loss = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)
                critic_loss_epoch += critic_loss.item()

                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                opt_critic.step()
            
            
            """
            Generator Training

            The generator learns by:
            1. Trying to fool the discriminator to predict 1 for fake images.
            """

            gen_fake = critic(fake).reshape(-1)
            gen_loss = -torch.mean(gen_fake)
            gen_loss_epoch += gen_loss.item()
            
            generator.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            if batch_idx % 100 == 0:
                print(f"EPOCH[{epoch}/{EPOCH_NUM}], "
                    f"BATCH[{batch_idx}/{len(loader)}]"
                    f"CRITIC LOSS: {critic_loss:.2f}, "
                    f"GEN LOSS: {gen_loss:.2f}"
                )
                
                with torch.no_grad():
                    fake = generator(fixed_noise)

                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    writer_real.add_image("Real", img_grid_real, global_step=step)

                step += 1
                
        avg_disc_loss = critic_loss_epoch / len(loader)
        avg_gen_loss = gen_loss_epoch / len(loader)
        critic_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)
                    
        if epoch % 10 == 0:
            save_checkpoint(epoch, generator, critic, opt_gen, opt_critic)
            check_gradients(generator)
            check_gradients(critic)
                    
        save_fake_images(generator, fixed_noise, epoch)
                    
    plot_losses(critic_losses, gen_losses)


def gan():
    clear_directories()
    
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
            ),
        ]
    )
    
    dataset = datasets.MNIST(root="mnist_dataset/", transform=transform, train=True, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Critic in WGAN is equivalent to Discriminator in normal GAN
    # It goes by the name because it no longer outputs between [0, 1]
    critic = Discriminator(IMAGE_CHANNELS, CRITIC_FEATURES).to(DEVICE)
    initialize_weights(critic)

    generator = Generator(NOISE_DIM, IMAGE_CHANNELS, GENERATOR_FEATURES).to(DEVICE)
    initialize_weights(generator)
    
    opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE, betas=(0.0, 0.9))
    opt_gen = optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.0, 0.9))
    # criterion = nn.BCELoss()  # Used to measure how well it distinguishes real from fake

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE)

    writer_fake = SummaryWriter(f"{LOG_DIR}/fake")
    writer_real = SummaryWriter(f"{LOG_DIR}/real")
    writer_loss = SummaryWriter(f"{LOG_DIR}/losses")
    
    # train_dcgan(discriminator, generator, fixed_noise, loader, opt_critic, opt_gen, criterion, writer_fake, writer_real, writer_loss)
    train_wgan(critic, generator, fixed_noise, loader, opt_critic, opt_gen, writer_fake, writer_real, writer_loss)

    create_gif()


if __name__ == "__main__":
    gan()