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

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2,
                                   padding=1)  # Input: (3, 96, 96) -> Output: (32, 48, 48)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (32, 48, 48) -> (64, 24, 24)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (64, 24, 24) -> (128, 12, 12)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (128, 12, 12) -> (256, 6, 6)
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_size)

        # Decoder
        self.dec_fc = nn.Linear(latent_size, 256 * 6 * 6)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.dec_conv5 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2, padding=2)
        # self.dec_conv6 = nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=2)

    def reparameterize_mean(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps*logvar)
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 6, 6)  # Reshape to match the beginning shape of the decoder
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        # x = F.relu(self.dec_conv5(x))
        # x = torch.sigmoid(self.dec_conv6(x))  # Ensure the output is in [0, 1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize_mean(mu, logvar)
        return self.decode(z), mu, logvar


# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


class Vae_Visualizer:
    def __init__(self, latent_size, vae, vae_cmp, initial_image_path, device="cpu", dist_method="cosine"):
        # Initialize the latent size and device
        self.latent_size = latent_size
        self.device = device
        self.vae = vae
        self.vae_cmp = vae_cmp
        self.init_latent = None
        self.dist_method = dist_method

        # Grid with 9 values [1,3]
        # Convert 2 latent space into image
        # Image into 8 latent
        # Store 8 latent in 2d array
        # Compute distance to slider values
        # Put distances into each pixel.
        self.latent_grid = [[0] * 10 for i in range(10)]
        self.image_grid = [[0] * 10 for i in range(10)]
        self.heatmap_grid = np.ones((10, 10, 3))
        self.img_diff_heatmap = np.ones((10, 10, 3))

        # Generate a placeholder image
        self.image = np.ones((1, 1, 3))

        # Create the figure for sliders
        self.fig_sliders, self.ax_sliders = plt.subplots()
        self.fig_sliders.canvas.manager.set_window_title("Sliders")
        self.ax_sliders.axis('off')

        # Create the figure for current image display
        self.fig_image, self.ax_image = plt.subplots()
        self.fig_image.canvas.manager.set_window_title("VAE Generated Image")
        self.img_display = self.ax_image.imshow(self.image, vmin=0, vmax=1)

        # Create the figure for current image display
        self.fig_image2, self.ax_image = plt.subplots()
        self.fig_image2.canvas.manager.set_window_title("Gym Generated Image")
        self.img_display2 = self.ax_image.imshow(self.image, vmin=0, vmax=1)

        # Create the figure for diff display
        self.fig_diff, self.ax_image = plt.subplots()
        self.fig_diff.canvas.manager.set_window_title("Diff Generated Image")
        x_range = 50 - (-40)  # 90
        y_range = 1 - (-0.8)  # 1.8
        aspect_ratio = x_range / y_range / (16 / 9)  # Adjust the division factor to fit the aspect of your display

        self.diff_display = self.ax_image.imshow(self.heatmap_grid, extent=[-40, 50, -1, 1], vmin=1,
                                                 aspect=aspect_ratio)

        # Set custom ticks for the x and y axes
        self.ax_image.set_yticks(np.arange(1, -1, -0.2))
        self.ax_image.set_xticks(np.arange(-40, 50, 10))

        # Set axis labels
        self.ax_image.set_xlabel('Angle')
        self.ax_image.set_ylabel('Position')

        # self.fig_img_diff, self.ax_image = plt.subplots()
        # self.fig_img_diff.canvas.manager.set_window_title("Image Diff Generated Image")
        # self.img_diff_display = self.ax_image.imshow(self.image_grid, vmin=0, vmax=1)

        # Initialize sliders
        self.sliders = []
        self.slider_values = [0] * latent_size

        # pos = -1.1
        # angle = -40
        pos = -1.2
        for i in range(10):
            pos += 0.2
            angle = -50
            for j in range(10):
                angle += 10
                print(f"{i} {j} {pos} {angle}")
                l2out = self.getImage(pos, angle)
                mu, logvar = vae.encode(l2out)
                latent8 = vae.reparameterize_mean(mu, logvar)
                self.latent_grid[i][j] = latent8
                self.image_grid[i][j] = l2out


        l2out = self.getImage(0, 0)
        mu, logvar = vae.encode(l2out)
        latent8 = vae.reparameterize_mean(mu, logvar)
        img_latent = latent8.squeeze(0)

        # # # Initalize with image
        # image = Image.open(initial_image_path)
        # image_array = np.array(image)
        # image_array = image_array / 255
        # # Transpose the image array to match the desired shape (3, 96, 96)
        # image_array = np.transpose(image_array, (2, 0, 1))
        # # Convert the numpy array to a TensorFlow tensor
        # image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        # print(image_tensor.shape)
        # mu, logvar = vae.encode(image_tensor)
        # img_latent = vae.reparameterize(mu, logvar)
        # img_latent = img_latent.squeeze(0)
        # print(image_tensor.shape)

        print(img_latent)
        for i in range(latent_size):
            self.slider_values[i] = img_latent.detach().numpy()[i]

        self.initialize_sliders()

        # Show both figures
        plt.show()

    def getImage(self, position=0, angle=30):
        n = 1  # Number of iterations

        # Create the environment
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.action_space.seed(82)

        observation, info = env.reset(seed=82)

        # Manually set the state of the environment

        env.env.env.env.state = np.array([position, 0, math.radians(angle), 0])

        # Take a random action
        action = env.action_space.sample()

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        img = env.render()

        # Convert the image to a PIL image
        img_pil = Image.fromarray(img)

        # Resize the image to 96x96 pixels
        img_resized = img_pil.resize((96, 96))

        # # Display the resized image
        # plt.imshow(img_resized)
        # plt.show()

        # Close the environment
        env.close()
        image_array = np.array(img_resized)
        image_array = image_array / 255
        # Transpose the image array to match the desired shape (3, 96, 96)
        image_array = np.transpose(image_array, (2, 0, 1))
        # Convert the numpy array to a TensorFlow tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

        return image_tensor

    def ned_torch(self, x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
        if len(x1.size()) == 1:
            ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
        elif x1.size() == torch.Size([x1.size(0), 1]):
            ned_2 = 0.5 * ((x1 - x2) ** 2 / (
                    x1 ** 2 + x2 ** 2 + eps)).squeeze()
        else:
            ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
        return ned_2 ** 0.5

    def nes_torch(self, x1, x2, dim=1, eps=1e-8):
        return 1 - self.ned_torch(x1, x2, dim, eps)

    def cosine_similarity(self, vec1, vec2):
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    def cosine_simmilarity_imgs(self, vec1, vec2):
        # Flatten the tensors (remove channel dimension)
        flattened_tensor1 = vec1.view(vec1.size(0), -1)
        flattened_tensor2 = vec2.view(vec2.size(0), -1)

        # Calculate cosine similarity (consider reducing along batch dimension if needed)
        similarity = F.cosine_similarity(flattened_tensor1, flattened_tensor2)
        average_similarity = torch.mean(similarity, dim=0)

        return average_similarity
        # Print the similarity value (closer to 1 indicates higher similarity)
    def mse(self, img1, img2):
        img1_flat = img1.reshape(-1)
        img2_flat = img2.reshape(-1)

        # Calculate the mean squared error
        mse = F.mse_loss(img1_flat, img2_flat)
        return mse

    def initialize_sliders(self):

        latent_vector_init = torch.tensor(self.slider_values, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Decode and set image
        img_init_tensor = self.vae.decode(latent_vector_init).squeeze(0).to(self.device)

        img_init = self.vae.decode(latent_vector_init).cpu().detach().numpy()
        img_init = img_init.squeeze(0)
        img_init = np.transpose(img_init, (1, 2, 0))

        self.img_display.set_data(img_init)
        self.fig_image.canvas.draw_idle()

        img_g = self.image_grid[5][4].squeeze(0).detach().numpy()
        img_g = np.transpose(img_g, (1, 2, 0))
        out, mu, logv = vae.forward(self.image_grid[5][4])
        out = np.transpose(out.cpu().detach().numpy().squeeze(0), (1, 2, 0))

        self.img_display2.set_data(out)
        self.fig_image2.canvas.draw_idle()

        diff_vals  = []
        # Initialize heatmap values
        for i in range(10):
            for j in range(10):
                # Compute normalized distance between cell and current image
                latent_vector_sq = latent_vector_init.squeeze(0)
                latent_vector_cell_sq = self.latent_grid[i][j].squeeze(0)

                img_cell_sq = self.image_grid[i][j].squeeze(0)

                if self.dist_method != "euclid":
                    dist = self.cosine_similarity(latent_vector_sq, latent_vector_cell_sq)
                    if dist < 0:
                        dist = 0
                else:
                    dist = self.nes_torch(x1=latent_vector_sq, x2=latent_vector_cell_sq,
                                          dim=latent_sz).detach().numpy()

                imgdist = self.cosine_simmilarity_imgs(img_init_tensor, img_cell_sq)
                # print(f"{i} {j} {self.mse(self.image_grid[0][0].squeeze(0), img_cell_sq)}")


                self.heatmap_grid[i][j][0] = self.heatmap_grid[i][j][1] = self.heatmap_grid[i][j][2] = dist
                diff_vals.append(1 - imgdist.item())

                self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][
                    2] = 1 - imgdist

        # Convert the list to a NumPy array
        arr = np.array(diff_vals)
        vector_length = np.linalg.norm(arr)
        normalized_arr = arr / vector_length

        ind = 0
        for i in range(10):
            for j in range(10):

                self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][
                    2] = normalized_arr[ind]
                # print(f"{i} {j} {normalized_arr[ind]}")
                # print(f"{i} {j} {diff_vals[ind]}")

                ind += 1


        self.diff_display.set_data(self.heatmap_grid)
        self.fig_diff.canvas.draw_idle()

        # self.img_diff_display.set_data(self.img_diff_heatmap)
        # self.fig_img_diff.canvas.draw_idle()

        # Stack sliders
        num_sliders = self.latent_size
        for i in range(num_sliders):
            ax_slider = self.fig_sliders.add_axes(
                [0.25, 0.95 - (i + 1) * 0.9 / num_sliders, 0.65, 0.9 / num_sliders * 0.8],
                facecolor='lightgoldenrodyellow')
            slider = Slider(ax_slider, f'Slider {i + 1}', -3, 3, valinit=self.slider_values[i])
            self.sliders.append(slider)

            def update(val, i=i):  # default argument to create closure
                # Set latent vector to slider values
                self.slider_values[i] = self.sliders[i].val
                print(self.slider_values)
                latent_vector = torch.tensor(self.slider_values, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Decode and set image
                img_cur_tensor = self.vae.decode(latent_vector).squeeze(0).to(self.device)
                img = self.vae.decode(latent_vector).cpu().detach().numpy()
                img = img.squeeze(0)
                img = np.transpose(img, (1, 2, 0))

                self.img_display.set_data(img)
                self.fig_image.canvas.draw_idle()

                diff_vals = []
                # Update heatmap values
                for i in range(10):
                    for j in range(10):
                        # Compute normalized distance between cell and current image
                        latent_vector_sq = latent_vector.squeeze(0)
                        latent_vector_cell_sq = self.latent_grid[i][j].squeeze(0)

                        if self.dist_method != "euclid":
                            dist = self.cosine_similarity(latent_vector_sq, latent_vector_cell_sq)
                            if dist < 0:
                                dist = 0
                        else:
                            dist = self.nes_torch(x1=latent_vector_sq, x2=latent_vector_cell_sq,
                                                  dim=latent_sz).detach().numpy()

                        img_cell_sq = self.image_grid[i][j].squeeze(0)

                        imgdist = self.mse(img_cur_tensor, img_cell_sq)


                        diff_vals.append(1 - imgdist.item())
                        self.heatmap_grid[i][j][0] = self.heatmap_grid[i][j][1] = self.heatmap_grid[i][j][2] = dist

                # Convert the list to a NumPy array
                arr = np.array(diff_vals)
                vector_length = np.linalg.norm(arr)
                normalized_arr = arr / vector_length

                ind = 0
                for i in range(10):
                    for j in range(10):
                        self.img_diff_heatmap[i][j][0] = self.img_diff_heatmap[i][j][1] = self.img_diff_heatmap[i][j][
                            2] = normalized_arr[ind]
                        ind += 1

                self.diff_display.set_data(self.heatmap_grid)
                self.fig_diff.canvas.draw_idle()

                # self.img_diff_display.set_data(self.img_diff_heatmap)
                # self.fig_img_diff.canvas.draw_idle()

            slider.on_changed(update)


if __name__ == '__main__':
    device = "cpu"
    latent_sz = 32
    vae = VAE(latent_size=latent_sz).to(device)
    # Load model weights
    best = torch.load("real_large_best_model.pth", map_location=torch.device('cpu'))
    vae.load_state_dict(best)
    vae.eval()

    latent_sz_cmp = 2
    vae_cmp = VAE(latent_size=latent_sz_cmp).to(device)
    best_cmp = torch.load("best_model_2.pth", map_location=torch.device('cpu'))
    vae_cmp.load_state_dict(best_cmp)
    vae_cmp.eval()

    viz = Vae_Visualizer(latent_sz, vae, vae_cmp, 'input_0.png', device)
