import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
#from utils.visualiser import Visualiser
#from utils.error_logger import ErrorLogger
from tensorboardX import SummaryWriter

from models import get_model

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset("HDF5")
    ds_path  = get_dataset_path("HDF5", json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    #test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=train_opts.batchSizeTrain, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=train_opts.batchSizeVal, shuffle=False)
    #test_loader  = DataLoader(dataset=test_dataset,  num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    #error_logger = ErrorLogger()
    writer = SummaryWriter(log_dir=model.save_dir)

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        train_loss_total = 0.0
        num_steps = 0
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # # Error visualisation
            # errors = model.get_current_errors()
            # error_logger.update(errors, split='train')
            
            #tensorboard loss 
            train_loss_total+= model.get_loss()
            num_steps += 1
            
        # tensorboard train loss
        train_loss_total_avg = train_loss_total / num_steps
        
        # Validation and Testing Iterations
        val_loss_total = 0.0
        num_steps = 0
        #for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
        for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
            split = 'validation'

            # Make a forward pass with the model
            model.set_input(images, labels)
            model.validate()

            # # Error visualisation
            # errors = model.get_current_errors()
            # stats = model.get_segmentation_stats()
            # error_logger.update({**errors, **stats}, split=split)

            # # Visualise predictions
            # visuals = model.get_current_visuals()
            # visualizer.display_current_results(visuals, epoch=epoch, save_result=False)
            
            #tensorboard loss
            val_loss_total += model.get_loss()
            num_steps += 1
        # tensorboard val loss 
        val_loss_total_avg = val_loss_total / num_steps

        # # Update the plots
        # for split in ['train', 'validation']:#, 'test']:
            # #visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            # visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        # error_logger.reset()
        
        # Visualize progress in tensorboard
        writer.add_scalars('losses', {
                                'val_loss': val_loss_total_avg,
                                'train_loss': train_loss_total_avg
                            }, epoch)
        lr = model.optimizers[0].param_groups[0]['lr']
        writer.add_scalar('learning_rate', lr, epoch)
        
        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
