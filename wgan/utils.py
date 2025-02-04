import os
import torch
import torch.nn as nn
import shutil
import imageio
import torchvision
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "checkpoints"
ASSETS_DIR = "assets"
IMAGE_DIR = "fake_images"
LOG_DIR = "logs"
directories=[MODEL_DIR, ASSETS_DIR, IMAGE_DIR, LOG_DIR]


def initialize_weights(model):
    """
    Initializes the weights of the model using a normal distribution.

    Args:
        model (torch.nn.Module): The neural network model whose weights need initialization.

    Note:
        - Applies only to convolutional layers (Conv2d, ConvTranspose2d) and BatchNorm layers.
        - Weights are initialized from a normal distribution with mean=0 and std=0.02.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(critic, real, fake):
    """
    Computes the gradient penalty to enforce the Lipschitz constraint in WGAN-GP.

    Args:
        critic (torch.nn.Module): The critic (discriminator) model.
        real (torch.Tensor): A batch of real images.
        fake (torch.Tensor): A batch of generated (fake) images.

    Returns:
        torch.Tensor: The gradient penalty value.

    Steps:
        1. Generates interpolated images by linearly mixing real and fake images using random alpha.
        2. Passes the interpolated images through the critic to obtain scores.
        3. Computes gradients of critic scores w.r.t. interpolated images.
        4. Reshapes gradients and calculates their L2 norm.
        5. Computes the gradient penalty as the mean squared difference from 1.
    """
    BATCH_SIZE, C, H, W = real.shape
    
    # Sample random weights (alpha) for interpolation
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(DEVICE)

    # Create interpolated images between real and fake
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Compute critic scores for interpolated images
    mixed_scores = critic(interpolated_images)

    # Compute gradients of critic scores with respect to interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Reshape gradient tensor
    gradient = gradient.view(gradient.shape[0], -1)

    # Compute L2 norm of gradients
    gradient_norm = gradient.norm(2, dim=1)

    # Compute gradient penalty (penalizing deviation from norm=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty


def clear_directories():
    """Helper function to clear directories."""
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


def check_gradients(model, threshold=1e-6):
    """Helper function to check weight gradients.

    Args:
        model (torch.nn.Module): The neural network model.
        threshold (float, optional): Minimum gradient value to detect vanishing gradients. Defaults to 1e-6.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            grad_std = param.grad.std().item()

            print(f"{name}: Mean={grad_mean:.6e}, Min={grad_min:.6e}, Max={grad_max:.6e}, Std={grad_std:.6e}")

            if abs(grad_mean) < threshold:
                print(f"⚠️ Warning: Possible vanishing gradient detected in {name} (mean={grad_mean:.6e})")
            elif abs(grad_max) > 1e3:  # Arbitrary high threshold to detect exploding gradients
                print(f"🚨 Warning: Possible exploding gradient detected in {name} (max={grad_max:.6e})")


def check_dead_neurons(layer, inputs):
    """Helper function to check dead neurons in ReLU layer.

    Args:
        layer (ReLU): ReLU layer.
        inputs (Tensor): A Tensor containing number of samples and neurons.
        
    Usage:
        relu = torch.nn.ReLU()
        inputs = torch.randn(100, 128)
        check_dead_neurons(relu, inputs)
        
    Output:
        If many neurons stay at zero (50%+), you likely have dead neurons.
    """
    outputs = layer(inputs)
    dead_neurons = (outputs == 0).sum().item()
    total_neurons = outputs.numel()
    print(f"Dead neurons: {dead_neurons}/{total_neurons} ({(dead_neurons/total_neurons)*100:.2f}%)")


def plot_losses(disc_losses, gen_losses, dir=ASSETS_DIR, filename="gan_loss.png"):
    """Helper function to visualize discriminator losses and generator losses.

    Args:
        disc_losses (list): Discriminator losses.
        gen_losses (list): Generator losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Training Loss")
    
    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/{filename}")
    
    plt.show()


def save_fake_images(generator, fixed_noise, epoch, dir=IMAGE_DIR):
    """Helper function to save fake images generated by the generator.

    Args:
        generator (Generator): The generator model responsible for creating fake images.
        fixed_noise (Tensor): A tensor of fixed random noise to generate images at specific epochs.
        epoch (int): The current epoch during training, used to differentiate saved images.
        save_dir (str, optional): Directory to save the generated images. Defaults to "images".
    """
    os.makedirs(dir, exist_ok=True)
    with torch.no_grad():
        fake = generator(fixed_noise)
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        torchvision.utils.save_image(img_grid, f"{dir}/epoch_{epoch}.png")


def create_gif(assets_dir=ASSETS_DIR, image_dir=IMAGE_DIR, filename="gan_training.gif"):
    """Creates a GIF from generated images stored in a directory.

    Args:
        image_dir (str, optional): Directory containing the saved images. Defaults to "images".
        output_gif (str, optional): The output GIF file name. Defaults to "gan_training.gif".
    """
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
    gif_images = [imageio.imread(img) for img in images]
    imageio.mimsave(f"{assets_dir}/{filename}", gif_images, fps=5)


def save_checkpoint(epoch, generator, discriminator, gen_opt, disc_opt, dir=MODEL_DIR):
    """Saves the model checkpoint to resume training from the same state.

    Args:
        epoch (int): The current epoch, used to name the checkpoint.
        generator (Generator): The generator model to save.
        discriminator (Discriminator): The discriminator model to save.
        gen_opt (torch.optim.Optimizer): The optimizer for the generator to save.
        disc_opt (torch.optim.Optimizer): The optimizer for the discriminator to save.
        save_path (str, optional): Directory to save the checkpoint. Defaults to "checkpoints".
    """
    os.makedirs(dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_opt_state_dict': gen_opt.state_dict(),
        'disc_opt_state_dict': disc_opt.state_dict(),
    }, f"{dir}/epoch_{epoch}.pth")
