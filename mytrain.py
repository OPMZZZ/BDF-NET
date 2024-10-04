import time
from options.train_options import TrainOptions
from data.myDataloader import create_dataset, CustomDataset
from models import create_model
from util.visualizer import Visualizer
import torch
from tqdm import tqdm

if __name__ == '__main__':
    opt = TrainOptions().parse()  # Parse training options

    # Create the dataset based on options
    dataset = create_dataset(opt)
    dataset_size = len(dataset) * 4  # Get the number of images in the dataset (multiplied by 4)
    print('Number of training patients = %d, Total number of training images = %d' % (dataset_size, dataset_size * 560))

    # Create the model based on options
    model = create_model(opt)
    model.setup(opt)  # Setup the model (load weights, create schedulers, etc.)

    # Create a visualizer for displaying/saving images and plots
    visualizer = Visualizer(opt)
    total_iters = 0  # Total number of iterations across epochs

    # Outer loop for training through epochs
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # Record the start time of the epoch
        iter_data_time = time.time()  # Record the time for loading data
        epoch_iter = 0  # Reset the number of iterations for the current epoch
        visualizer.reset()  # Reset the visualizer to ensure at least one result is saved to HTML each epoch
        model.update_learning_rate()  # Update the learning rate at the start of each epoch

        # Inner loop through each patient in the dataset
        for _, patient in enumerate(dataset):
            minidataset = CustomDataset(patient)
            data_loader = torch.utils.data.DataLoader(minidataset, batch_size=opt.batch_size, shuffle=True,
                                                      num_workers=opt.num_threads)

            for i in range(1):  # Outer iteration loop
                for data in data_loader:  # Inner loop through the data
                    iter_start_time = time.time()  # Record the time for each iteration
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time  # Calculate data loading time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)  # Unpack and preprocess data from the dataset
                    model.optimize_parameters(i)  # Compute loss, get gradients, update network weights

                    # Display images on Visdom and save them to HTML every <display_freq> iterations
                    if total_iters % opt.display_freq == 0:
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    # Print training losses and save log information to disk every <print_freq> iterations
                    if total_iters % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / opt.batch_size  # Time per batch
                        visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(total_iters) / dataset_size, losses)

                    # Save the latest model every <save_latest_freq> iterations
                    if total_iters % opt.save_latest_freq == 0:
                        print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)

                iter_data_time = time.time()  # Record the time for the next iteration

        # Save the model at the end of every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('Saving model at the end of epoch %d, total_iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Print the time taken to complete the epoch
        print('End of epoch %d / %d \t Time Taken: %d seconds' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
