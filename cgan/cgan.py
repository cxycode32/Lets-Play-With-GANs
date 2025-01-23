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
CLASSES_NUM = 10
EMBEDDING_SIZE = 100
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
    def __init__(self, img_channels, disc_features, classes_num, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels+1, disc_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            self._block(disc_features, disc_features * 2, 4, 2, 1),
            self._block(disc_features * 2, disc_features * 4, 4, 2, 1),
            self._block(disc_features * 4, disc_features * 8, 4, 2, 1),

            nn.Conv2d(disc_features * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(classes_num, img_size * img_size)
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, img_channels, gen_features, classes_num, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            self._block(channels_noise + embed_size, gen_features * 16, 4, 1, 0),
            self._block(gen_features * 16, gen_features * 8, 4, 2, 1),
            self._block(gen_features * 8, gen_features * 4, 4, 2, 1),
            self._block(gen_features * 4, gen_features * 2, 4, 2, 1),

            nn.ConvTranspose2d(gen_features * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(classes_num, embed_size)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def train_wgan(critic, generator, fixed_noise, loader, opt_critic, opt_gen, writer_fake, writer_real, writer_loss):
    step = 0
    critic_losses, gen_losses = [], []
    
    critic.train()
    generator.train()
    
    for epoch in range(EPOCH_NUM):
        critic_loss_epoch = 0.0
        gen_loss_epoch = 0.0
        
        for batch_idx, (real, labels) in enumerate(tqdm(loader)):

            """
            Critic Training

            The critic learns by:
            1. Getting real images -> predict 1
            2. Getting fake images -> predict 0
            3. Update weights to improve accuracy
            """
            
            real, labels = real.to(DEVICE), labels.to(DEVICE)
            cur_batch_size = real.shape[0]
            
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, NOISE_DIM, 1, 1).to(DEVICE)
                fake = generator(noise, labels)  # Create fake images
            
                critic_real = critic(real, labels).reshape(-1)  # Critic prediction for real images
                critic_fake = critic(fake, labels).reshape(-1)  # Critic prediction for fake images

                gp = gradient_penalty(critic, labels, real, fake)

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

            gen_fake = critic(fake, labels).reshape(-1)
            gen_loss = -torch.mean(gen_fake)
            gen_loss_epoch += gen_loss.item()
            
            generator.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            if batch_idx % 100 == 0:
                print(f"EPOCH[{epoch + 1}/{EPOCH_NUM}], "
                    f"BATCH[{batch_idx}/{len(loader)}]"
                    f"CRITIC LOSS: {critic_loss:.2f}, "
                    f"GEN LOSS: {gen_loss:.2f}"
                )
                
                with torch.no_grad():
                    fake = generator(noise, labels)

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
                    
        save_fake_images(generator, labels, noise, epoch)
                    
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
    critic = Discriminator(IMAGE_CHANNELS, CRITIC_FEATURES, CLASSES_NUM, IMAGE_SIZE).to(DEVICE)
    initialize_weights(critic)

    generator = Generator(NOISE_DIM, IMAGE_CHANNELS, GENERATOR_FEATURES, CLASSES_NUM, IMAGE_SIZE, EMBEDDING_SIZE).to(DEVICE)
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