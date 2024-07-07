import math
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib.widgets import Slider

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # Encoder layers: convolutional layers to encode the input image to a latent representation
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # Input: (3, 96, 96) -> Output: (32, 48, 48)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (32, 48, 48) -> (64, 24, 24)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (64, 24, 24) -> (128, 12, 12)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (128, 12, 12) -> (256, 6, 6)
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)  # Fully connected layer for mean of the latent space
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_size)  # Fully connected layer for log variance of the latent space

        # Decoder layers: deconvolutional layers to decode the latent representation back to an image
        self.dec_fc = nn.Linear(latent_size, 256 * 6 * 6)  # Fully connected layer to map latent vector to intermediate representation
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def reparameterize_mean(self, mu, logvar):
        # Use mean value for reparameterization
        return mu

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * logvar)

    def encode(self, x):
        # Encode input image to latent space
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        # Decode latent space to output image
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 6, 6)  # Reshape to match the beginning shape of the decoder
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        return x

    def forward(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize_mean(mu, logvar)
        return self.decode(z), mu, logvar

# Define the loss function for the VAE
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
class Vae_Visualizer:
    def __init__(self, latent_size, vae, initial_image_path, device="cpu", dist_method="cosine"):
        # Initialize the visualizer with VAE models and other parameters
        self.latent_size = latent_size  # Size of the latent vector
        self.device = device  # Device to use for computation (CPU or GPU)
        self.vae = vae  # VAE model
        self.init_latent = None  # Placeholder for initial latent vector
        self.dist_method = dist_method  # Distance method for comparison (cosine by default)

        # Initialize grids for latent vectors, images, and heatmaps
        self.latent_grid = [[0] * 10 for i in range(10)]  # Grid to store latent vectors
        self.image_grid = [[0] * 10 for i in range(10)]  # Grid to store images
        self.heatmap_grid = np.ones((10, 10, 3))  # Heatmap grid for similarity visualization
        self.img_diff_heatmap = np.ones((10, 10, 3))  # Heatmap grid for image differences

        # Generate a placeholder image
        self.image = np.ones((1, 1, 3))  # Placeholder image for display

        # Create the figure for sliders
        self.fig_sliders, self.ax_sliders = plt.subplots()  # Figure for sliders
        self.fig_sliders.canvas.manager.set_window_title("Sliders")  # Set window title for sliders
        self.ax_sliders.axis('off')  # Turn off axis for sliders

        # Create the figure for current image display
        self.fig_image, self.ax_image = plt.subplots()  # Figure for VAE generated image
        self.fig_image.canvas.manager.set_window_title("VAE Generated Image")  # Set window title for VAE image
        self.img_display = self.ax_image.imshow(self.image, vmin=0, vmax=1)  # Display placeholder image

        # Create the figure for gym generated image display
        self.fig_image2, self.ax_image2 = plt.subplots()  # Figure for gym generated image
        self.fig_image2.canvas.manager.set_window_title("Gym Generated Image")  # Set window title for gym image
        self.img_display2 = self.ax_image2.imshow(self.image, vmin=0, vmax=1)  # Display placeholder image

        # Create the figure for difference heatmap display
        self.fig_diff, self.ax_diff = plt.subplots()  # Figure for difference heatmap
        self.fig_diff.canvas.manager.set_window_title("Diff Generated Image")  # Set window title for diff image

        # Calculate aspect ratio for the heatmap
        x_range = 50 - (-40)  # Range of angles
        y_range = 1 - (-0.8)  # Range of positions
        aspect_ratio = x_range / y_range / (16 / 9)  # Adjust aspect ratio to fit display

        # Display heatmap grid for differences
        self.diff_display = self.ax_diff.imshow(self.heatmap_grid, extent=[-40, 50, -1, 1], vmin=1, aspect=aspect_ratio)  # Show the heatmap

        # Set custom ticks for the x and y axes
        self.ax_diff.set_yticks(np.arange(1, -1, -0.2))  # Custom y-axis ticks
        self.ax_diff.set_xticks(np.arange(-40, 50, 10))  # Custom x-axis ticks

        # Set axis labels
        self.ax_diff.set_xlabel('Angle')  # Label for x-axis
        self.ax_diff.set_ylabel('Position')  # Label for y-axis

        # Initialize sliders
        self.sliders = []  # List to store sliders
        self.slider_values = [0] * latent_size  # Initial slider values

        # Initialize grid and latent values with image data
        pos = -1.2  # Initial position
        for i in range(10):
            pos += 0.2  # Increment position
            angle = -50  # Initial angle
            for j in range(10):
                angle += 10  # Increment angle
                print(f"{i} {j} {pos} {angle}")  # Print grid indices, position, and angle
                l2out = self.getImage(pos, angle)  # Get image tensor for the given position and angle
                mu, logvar = vae.encode(l2out)  # Encode image to get mean and log variance
                latent8 = vae.reparameterize_mean(mu, logvar)  # Reparameterize to get latent vector
                self.latent_grid[i][j] = latent8  # Store latent vector in grid
                self.image_grid[i][j] = l2out  # Store image tensor in grid

        l2out = self.getImage(0, 0)  # Get image tensor for position 0 and angle 0
        mu, logvar = vae.encode(l2out)  # Encode image to get mean and log variance
        latent8 = vae.reparameterize_mean(mu, logvar)  # Reparameterize to get latent vector
        img_latent = latent8.squeeze(0)  # Squeeze the latent vector to remove batch dimension

        # Set initial slider values based on the latent vector from the initial image
        print(img_latent)  # Print the initial latent vector
        for i in range(latent_size):
            self.slider_values[i] = img_latent.detach().numpy()[i]  # Set slider value to the corresponding latent vector value

        self.initialize_sliders()  # Initialize sliders with the values

        # Show all figures
        plt.show()  # Display the figures

    def getImage(self, position=0, angle=30):
        # Generate image from CartPole environment based on position and angle
        n = 1  # Number of iterations

        # Create the environment
        env = gym.make("CartPole-v1", render_mode="rgb_array")  # Create CartPole environment with RGB rendering
        env.action_space.seed(82)  # Seed the action space for reproducibility

        observation, info = env.reset(seed=82)  # Reset environment and get initial observation

        # Manually set the state of the environment
        env.env.env.env.state = np.array([position, 0, math.radians(angle), 0])  # Set state based on position and angle

        # Take a random action
        action = env.action_space.sample()  # Sample a random action

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)  # Step the environment

        # Render the environment
        img = env.render()  # Render the environment to get an image

        # Convert the image to a PIL image
        img_pil = Image.fromarray(img)  # Convert to PIL image

        # Resize the image to 96x96 pixels
        img_resized = img_pil.resize((96, 96))  # Resize the image

        # Close the environment
        env.close()  # Close the environment

        # Convert the resized image to a numpy array and normalize
        image_array = np.array(img_resized) / 255.0  # Normalize image to range [0, 1]

        # Transpose the image array to match the desired shape (3, 96, 96)
        image_array = np.transpose(image_array, (2, 0, 1))  # Transpose to (3, 96, 96)

        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dimension

        return image_tensor  # Return the image tensor

    def ned_torch(self, x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
        # Calculate normalized Euclidean distance (NED) between two tensors
        if len(x1.size()) == 1:  # Check if tensors are 1D
            ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))  # Calculate NED for 1D tensors
        elif x1.size() == torch.Size([x1.size(0), 1]):  # Check if tensors are column vectors
            ned_2 = 0.5 * ((x1 - x2) ** 2 / (x1 ** 2 + x2 ** 2 + eps)).squeeze()  # Calculate NED for column vectors
        else:
            ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))  # Calculate NED for higher-dimensional tensors
        return ned_2 ** 0.5  # Return the square root of NED

    def nes_torch(self, x1, x2, dim=1, eps=1e-8):
        # Calculate normalized Euclidean similarity (NES) between two tensors
        return 1 - self.ned_torch(x1, x2, dim, eps)  # NES is 1 minus NED

    def cosine_similarity(self, vec1, vec2):
        # Calculate cosine similarity between two vectors
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()  # Return cosine similarity

    def cosine_simmilarity_imgs(self, vec1, vec2):
        # Calculate cosine similarity between two image tensors
        flattened_tensor1 = vec1.view(vec1.size(0), -1)  # Flatten the first tensor
        flattened_tensor2 = vec2.view(vec2.size(0), -1)  # Flatten the second tensor

        # Calculate cosine similarity (consider reducing along batch dimension if needed)
        similarity = torch.nn.functional.cosine_similarity(flattened_tensor1, flattened_tensor2)  # Calculate cosine similarity
        average_similarity = torch.mean(similarity, dim=0)  # Calculate mean similarity

        return average_similarity  # Return average similarity

    def mse(self, img1, img2):
        # Calculate mean squared error (MSE) between two images
        img1_flat = img1.reshape(-1)  # Flatten the first image
        img2_flat = img2.reshape(-1)  # Flatten the second image

        # Calculate the mean squared error
        mse = torch.nn.functional.mse_loss(img1_flat, img2_flat)  # Compute MSE
        return mse  # Return MSE

    def initialize_sliders(self):
        # Initialize sliders and their callbacks
        latent_vector_init = torch.tensor(self.slider_values, dtype=torch.float32).unsqueeze(0).to(self.device)  # Convert slider values to tensor

        # Decode and set initial image
        img_init_tensor = self.vae.decode(latent_vector_init).squeeze(0).to(self.device)  # Decode initial latent vector
        img_init = self.vae.decode(latent_vector_init).cpu().detach().numpy()  # Convert to numpy array
        img_init = img_init.squeeze(0)  # Squeeze to remove batch dimension
        img_init = np.transpose(img_init, (1, 2, 0))  # Transpose to (H, W, C)

        self.img_display.set_data(img_init)  # Set initial image data
        self.fig_image.canvas.draw_idle()  # Update figure

        # Initialize grid image
        img_g = self.image_grid[5][4].squeeze(0).detach().numpy()  # Get initial grid image
        img_g = np.transpose(img_g, (1, 2, 0))  # Transpose to (H, W, C)
        out, mu, logv = self.vae.forward(self.image_grid[5][4])  # Encode and decode the grid image
        out = np.transpose(out.cpu().detach().numpy().squeeze(0), (1, 2, 0))  # Convert to numpy array and transpose

        self.img_display2.set_data(out)  # Set grid image data
        self.fig_image2.canvas.draw_idle()  # Update figure

        # Initialize heatmap values
        diff_vals = []  # List to store difference values
        for i in range(10):
            for j in range(10):
                # Compute normalized distance between cell and current image
                latent_vector_sq = latent_vector_init.squeeze(0)  # Squeeze the initial latent vector
                latent_vector_cell_sq = self.latent_grid[i][j].squeeze(0)  # Squeeze the cell latent vector
                img_cell_sq = self.image_grid[i][j].squeeze(0)  # Squeeze the cell image

                if self.dist_method != "euclid":  # Check if distance method is not Euclidean
                    dist = self.cosine_similarity(latent_vector_sq, latent_vector_cell_sq)  # Calculate cosine similarity
                    if dist < 0:  # Ensure non-negative similarity
                        dist = 0
                else:
                    dist = self.nes_torch(x1=latent_vector_sq, x2=latent_vector_cell_sq, dim=self.latent_size).detach().numpy()  # Calculate NES

                imgdist = self.cosine_simmilarity_imgs(img_init_tensor, img_cell_sq)  # Calculate image similarity

                self.heatmap_grid[i][j][0] = self.heatmap_grid[i][j][1] = self.heatmap_grid[i][j][2] = dist  # Set heatmap values
                diff_vals.append(1 - imgdist.item())  # Append difference value

                self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][2] = 1 - imgdist  # Set image difference heatmap values

        # Normalize the difference values for the heatmap
        arr = np.array(diff_vals)  # Convert list to numpy array
        vector_length = np.linalg.norm(arr)  # Calculate vector length
        normalized_arr = arr / vector_length  # Normalize the array

        ind = 0  # Index for normalized array
        for i in range(10):
            for j in range(10):
                self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][2] = normalized_arr[ind]  # Set normalized values
                ind += 1

        self.diff_display.set_data(self.heatmap_grid)  # Update heatmap data
        self.fig_diff.canvas.draw_idle()  # Redraw the figure

        # Initialize and stack sliders
        num_sliders = self.latent_size  # Number of sliders
        for i in range(num_sliders):
            # Create axis for each slider
            ax_slider = self.fig_sliders.add_axes([0.25, 0.95 - (i + 1) * 0.9 / num_sliders, 0.65, 0.9 / num_sliders * 0.8], facecolor='lightgoldenrodyellow')
            # Create slider
            slider = Slider(ax_slider, f'Slider {i + 1}', -3, 3, valinit=self.slider_values[i])
            self.sliders.append(slider)  # Append slider to list

            def update(val, i=i):  # Default argument to create closure
                # Update latent vector based on slider values
                self.slider_values[i] = self.sliders[i].val  # Update slider value
                print(self.slider_values)  # Print current slider values
                latent_vector = torch.tensor(self.slider_values, dtype=torch.float32).unsqueeze(0).to(self.device)  # Convert to tensor

                # Decode and set current image
                img_cur_tensor = self.vae.decode(latent_vector).squeeze(0).to(self.device)  # Decode latent vector
                img = self.vae.decode(latent_vector).cpu().detach().numpy()  # Convert to numpy array
                img = img.squeeze(0)  # Squeeze to remove batch dimension
                img = np.transpose(img, (1, 2, 0))  # Transpose to (H, W, C)

                self.img_display.set_data(img)  # Set current image data
                self.fig_image.canvas.draw_idle()  # Update figure

                # Update heatmap values
                diff_vals = []  # List to store difference values
                for i in range(10):
                    for j in range(10):
                        latent_vector_sq = latent_vector.squeeze(0)  # Squeeze the latent vector
                        latent_vector_cell_sq = self.latent_grid[i][j].squeeze(0)  # Squeeze the cell latent vector

                        if self.dist_method != "euclid":  # Check if distance method is not Euclidean
                            dist = self.cosine_similarity(latent_vector_sq, latent_vector_cell_sq)  # Calculate cosine similarity
                            if dist < 0:  # Ensure non-negative similarity
                                dist = 0
                        else:
                            dist = self.nes_torch(x1=latent_vector_sq, x2=latent_vector_cell_sq, dim=self.latent_size).detach().numpy()  # Calculate NES

                        img_cell_sq = self.image_grid[i][j].squeeze(0)  # Squeeze the cell image
                        imgdist = self.mse(img_cur_tensor, img_cell_sq)  # Calculate image MSE

                        diff_vals.append(1 - imgdist.item())  # Append difference value
                        self.heatmap_grid[i][j][0] = self.heatmap_grid[i][j][1] = self.heatmap_grid[i][j][2] = dist  # Set heatmap values

                # Normalize the difference values
                arr = np.array(diff_vals)  # Convert list to numpy array
                vector_length = np.linalg.norm(arr)  # Calculate vector length
                normalized_arr = arr / vector_length  # Normalize the array

                ind = 0  # Index for normalized array
                for i in range(10):
                    for j in range(10):
                        self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][2] = normalized_arr[ind]  # Set normalized values
                        ind += 1

                self.diff_display.set_data(self.heatmap_grid)  # Update heatmap data
                self.fig_diff.canvas.draw_idle()  # Redraw the figure

            slider.on_changed(update)  # Set update function for slider


if __name__ == '__main__':
    device = "cpu"
    latent_sz = 32
    vae = VAE(latent_size=latent_sz).to(device)
    # Load model weights
    best = torch.load("real_large_best_model.pth", map_location=torch.device('cpu'))
    vae.load_state_dict(best)
    vae.eval()


    viz = Vae_Visualizer(latent_sz, vae, 'input_0.png', device)
