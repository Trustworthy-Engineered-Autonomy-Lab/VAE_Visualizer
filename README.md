# VAE Visualization Tool

This project visualizes the output of a Variational Autoencoder (VAE) using sliders to manipulate the latent space. It includes a visualizer to show generated images and the differences between them.

The `Vae_Visualizer` class provides an interactive visualization tool for exploring the latent space of a Variational Autoencoder (VAE) using sliders. It takes an input image of the OpenAI CartPole environment, processes it into the latent space using the VAE, and allows users to manipulate the latent vectors via sliders. The generated images are displayed and compared to a heatmap grid of other CartPole images with various angles and positions, where darker cells indicate less similarity and lighter cells indicate more.
## Features

- **VAE Architecture:** Convolutional layers for encoding and decoding images.
- **Visualization:** Interactive sliders to manipulate the latent space and observe changes in the generated images.
- **Heatmap:** Display of cosine similarity heatmap for the latent vectors.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Trustworthy-Engineered-Autonomy-Lab/VAE_Visualizer
   cd VAE_Visualizer
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Input Files

1. **Model Weights:**
   - Ensure you have pre-trained VAE model weights saved as `.pth` files.
   - Place the model weights files in the project directory.

2. **Initial Image:**
   - Provide an initial image file (e.g., `input_0.png`).
   - The image will be used to initialize the latent space.

### Running the Visualizer

1. Update the model file paths and parameters in the `if __name__ == '__main__':` section of the code:

   ```python
   if __name__ == '__main__':
       device = "cpu"  # Device to run the VAE (cpu or cuda)
       latent_sz = 32  # Latent size for the primary VAE
       vae = VAE(latent_size=latent_sz).to(device)
       
       # Load model weights for the primary VAE
       best = torch.load("real_large_best_model.pth", map_location=torch.device('cpu'))
       vae.load_state_dict(best)
       vae.eval()



       # Initialize the visualizer with the latent sizes, VAEs, initial image, and device
       viz = Vae_Visualizer(latent_sz, vae, 'input_0.png', device)
   ```

2. Run the visualizer:

   ```bash
   python vae_visualizer.py
   ```

3. Interact with the visualizer:
   - **Sliders Window:** Adjust the sliders to manipulate the latent space. Each slider corresponds to a dimension in the latent vector.
   - **VAE Generated Image Window:** Observe the generated image from the VAE as you adjust the sliders.
   - **Gym Generated Image Window:** Compare the VAE generated image with the images from the CartPole environment.
   - **Diff Generated Image Window:** View the heatmap showing cosine similarity between the current latent vector and the grid cells.

### Parameters in Vae_Visualizer Class

1. **latent_size:** Size of the latent vector. Defines the dimensionality of the latent space.
2. **vae:** The VAE model used for generating images.
3. **initial_image_path:** Path to the initial image used to initialize the latent vector.
4. **device:** Device to run the VAE models (e.g., 'cpu' or 'cuda').
5. **dist_method:** Method to compute distance between latent vectors. Options include 'cosine' for cosine similarity and 'euclid' for Euclidean distance.

### Methods in Vae_Visualizer Class

1. **getImage(position, angle):** Generates an image from the CartPole environment given a specific position and angle.
2. **ned_torch(x1, x2, dim, eps):** Computes normalized Euclidean distance between two tensors.
3. **nes_torch(x1, x2, dim, eps):** Computes normalized Euclidean similarity between two tensors.
4. **cosine_similarity(vec1, vec2):** Computes cosine similarity between two vectors.
5. **cosine_simmilarity_imgs(vec1, vec2):** Computes cosine similarity between two images.
6. **mse(img1, img2):** Computes mean squared error between two images.
7. **initialize_sliders():** Initializes sliders for manipulating the latent space and sets up the initial display images and heatmaps.

### Example Workflow

1. **Setup:**
   - Place your VAE model weights and initial image in the project directory.
   - Update the file paths and parameters in the script.

2. **Run the Script:**
   - Execute `python vae_visualizer.py` to start the visualizer.

3. **Interact:**
   - Use the sliders to explore the latent space and observe changes in the generated images.
   - Compare the VAE generated images with the CartPole environment images.

## Dependencies

- Python 3.x
- PyTorch
- Gym
- Matplotlib
- NumPy
- PIL


## Acknowledgments

- This project uses the CartPole environment from OpenAI Gym.
