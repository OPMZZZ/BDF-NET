import os
from options.test_options import TestOptions
from data.myDataloader import create_dataset, CustomDataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import matplotlib.pyplot as plt

# Function to visualize and save results for comparison
def draw(real, img1, img2, slice):
    # Convert tensors to numpy arrays and select specific slices for visualization
    img1 = img1.cpu().numpy()[0, [0, 5, 10, 15, -1], :, :]
    img2 = img2.cpu().numpy()[0, [0, 5, 10, 15, -1], :, :]
    real1 = real.cpu()[0, 0, :, :].unsqueeze(0).numpy()
    real2 = real.cpu()[0, -1, :, :].unsqueeze(0).numpy()

    # Concatenate real and predicted images for comparison
    img1 = np.concatenate((real1, img1, real2), axis=0)
    img2 = np.concatenate((real1, img2, real2), axis=0)

    # Compute the absolute difference between the two sets of images
    img3 = np.abs(img1 - img2)

    # Create a 3x7 grid of subplots for visualization
    fig, axes = plt.subplots(3, 7, figsize=(26, 8))

    # Organize images into lists for easier iteration
    imgs = [img1, img2]
    name = ['real', 'fake']

    # Loop through real and fake images to display them on the subplots
    for i in range(2):
        img = imgs[i]
        for j in range(7):
            img1 = img[j]
            # Display the image on the subplot
            axes[i, j].imshow(img1, cmap='binary')
            # Hide axis labels
            axes[i, j].axis('off')

    # Display the difference images in the last row with a color bar
    for j in range(7):
        im = axes[2, j].imshow(img3[j], cmap='jet')
        fig.colorbar(im, ax=axes[2, j])
        axes[2, j].axis('off')

    # Adjust layout and save the image to disk
    plt.tight_layout()
    plt.savefig(f'./outputs/{slice:03d}.png')
    plt.close()

if __name__ == '__main__':
    opt = TestOptions().parse()  # Parse testing options

    # Override some options for testing
    opt.num_threads = 0  # Set number of threads to 0 as the test code only supports single-threading
    opt.batch_size = 1  # Set batch size to 1 for testing
    opt.serial_batches = True  # Disable shuffling for consistency during testing
    opt.no_flip = True  # Disable image flipping for testing
    opt.display_id = -1  # Disable Visdom display, results will be saved to HTML files instead

    # Create the dataset based on options
    dataset = create_dataset(opt, size=1)

    # Create the model based on options
    model = create_model(opt)

    # Loop through a few iterations (hardcoded as 1 to 2 here)
    for i in range(1, 2):
        opt.load_iter = i
        model.setup(opt)  # Setup the model by loading weights, printing the network structure, etc.

        # Create a directory for saving results
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
        if opt.load_iter > 0:  # Append iteration number to the directory name if not zero
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('Creating website directory', web_dir)

        # Initialize an HTML object to save the experiment's results
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        # Set model to evaluation mode if specified; affects layers like batchnorm and dropout
        if opt.eval:
            model.eval()

        # Initialize lists to store metrics
        mean_rmse = []
        mean_ssim = []
        mean_psnr = []

        # Loop through the dataset (outer loop through patients)
        for _, patient in enumerate(dataset):
            # Create a dataloader for the patient-specific data
            minidataset = CustomDataset(patient)
            data_loader = torch.utils.data.DataLoader(minidataset, batch_size=opt.batch_size,
                                                      num_workers=opt.num_threads)

            # Inner loop through the data
            for i, data in enumerate(data_loader):
                model.set_input(data)  # Unpack data from dataloader and prepare for inference
                model.test()  # Run inference

                # Get visuals and loss metrics from the model
                visuals = model.get_current_visuals()
                x, y, z, d = model.cal_loss()

                # Append metrics to their respective lists
                mean_rmse.append(x)
                mean_ssim.append(y)
                mean_psnr.append(z)

                # Optionally, visualize and save the images
                # draw(*d, i)
                # save_images(webpage, visuals, i, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

            # Save the model's state
            model.save()

            # Print the average metrics across the dataset
            print(f'RMSE: {np.mean(mean_rmse)}')
            print(f'SSIM: {np.mean(mean_ssim)}')
            print(f'PSNR: {np.mean(mean_psnr)}')

        # Save the results as an HTML webpage
        webpage.save()
