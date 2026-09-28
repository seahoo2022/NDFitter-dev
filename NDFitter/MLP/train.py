import torch
import torch.optim as optim
from torch.autograd import Variable
from NDFitter.MLP.model.model import FeedForwardNN
from NDFitter.MLP.data.dataloader import get_dataloader
import matplotlib.pyplot as plt
import argparse
from transformers import get_linear_schedule_with_warmup
import logging
import torch.nn.functional as F
import numpy as np
from NDFitter.utils import *
from NDFitter.paths import project_path
# from NDFitter.MLP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, criterion, optimizer, scheduler, num_epochs=100):
    writer = SummaryWriter(log_dir=project_path("outputs/tensorboard"))

    losses = []
    output_losses = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()
    total_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            total_step += 1
            inputs, targets = Variable(inputs), Variable(targets)
            inputs.requires_grad_()
            optimizer.zero_grad()
            outputs = model(inputs)
            if args.loss == 'partial_deriv':
                # print(inputs.requires_grad)
                output_grads = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
                grad_x = output_grads[:, 0].unsqueeze(1)

                loss = torch.abs(grad_x-targets)

                # output_grads2 = torch.autograd.grad(outputs=output_grads[:, 0], inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
                # grad_x2 = output_grads2[:, 0].unsqueeze(1)
                # weights = 1 + torch.tanh(grad_x2)  # Using tanh for scaling; can be adjusted

                weights = 1

                # Apply weights to loss
                weighted_loss = weights * loss

                # Optionally, apply a threshold to ignore minor deviations
                threshold = 0.1  # This value can be tuned
                weighted_loss = torch.where(weighted_loss < threshold, torch.tensor(0.0), weighted_loss)

                loss = torch.mean(weighted_loss)
                # loss = torch.mean((grad_x-targets) ** 2)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate for next step
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Training Loss', loss, total_step)
            writer.add_scalar('Learning Rate', current_lr, total_step)
            # print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr}")
            epoch_loss += loss.item()
            losses.append(np.log10(loss.item()))
            # if i % 100 == 0:  # update plot every 10 batches
            #     ax.clear()
            #     ax.plot(losses)
            #     plt.pause(0.001)
            
        avg_loss = epoch_loss / len(dataloader)
        
        
        output_losses.append(avg_loss)
    # plt.savefig('loss.png')
    # plt.ioff()
    writer.close()
    ax.plot(losses)
    return output_losses, fig

def main(args):
    # Define your model
    if args.act == 'sigmoid':
        activation = F.sigmoid
    elif args.act == 'relu':
        activation = F.relu
    elif args.act == 'tanh':
        activation = F.tanh
    elif args.act == 'leaky_relu':
        activation = F.leaky_relu
    elif args.act == 'elu':
        activation = F.elu
    elif args.act == 'selu':
        activation = F.selu
    elif args.act == 'gelu':
        activation = F.gelu
    else:
        activation = F.relu

    model = FeedForwardNN(args.layers, activation=activation)

    if args.load != None:
        loaded_model = torch.load(args.load)
        model.load_state_dict(loaded_model)
    # Get dataloaders
    train_loader, _ = get_dataloader(folder=args.folder, batch_size=args.batch_size)

    lambdax = args.lambdax

    # Define criterion and optimizer
    if args.loss == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'relativel2':
        criterion = relative_l2_error
    elif args.loss == 'relativel2_max':
        criterion = relative_l2_error_max
    elif args.loss == 'partial_deriv':
        def criterion(output,target):
            output_grads = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output), create_graph=True)[0]
            pass


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr,weight_decay=lambdax)

    # Define scheduler
    if args.scheduler == 'lambda':
            # Step 2 Define Optimizer and scheduler
        # define the number of warmup steps and total training steps
        num_total_steps = len(train_loader) * args.epochs
        num_warmup_steps = num_total_steps//10
        
        # define the warmup function
        def warmup_lambda(current_step):
            return 1.0
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        # define the decay function
        def decay_lambda(current_step):
            print('here?')
            return 1
            if current_step < num_total_steps//10:
                # return 0.5 ** (current_step // (num_total_steps // 3))
                return 0.7
            elif current_step < num_total_steps//5:
                return 0.5
            elif current_step < num_total_steps//2:
                return 0.3
            elif current_step < num_total_steps:
                return 0.1
            return 0.01
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: decay_lambda(step) * warmup_lambda(step))

    elif args.scheduler == 'linear':
        num_training_steps = len(train_loader) * args.epochs
        num_warmup_steps = num_training_steps // 10  # Warmup for the first 10% of training steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler == 'onecycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Train the model
    losses, loss_fig = train(model, train_loader, criterion, optimizer, scheduler, args.epochs)

    # Save the model
    if args.output == None:
        output = os.path.join(args.folder, now()+f'layers_{"_".join(map(str, args.layers))}_batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_act_{args.act}_wd_{args.lambdax}_split_{args.split}_loss_{args.loss}')
    else:
        output = args.output
    mkdir(output)
    model_path = f'{output}/model_layers_{"_".join(map(str, args.layers))}_batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_act_{args.act}_wd_{args.lambdax}_split_{args.split}_loss_{args.loss}.pt'
    torch.save(model.state_dict(), model_path)
    loss_fig.savefig(f'{output}/model_layers_{"_".join(map(str, args.layers))}_batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_act_{args.act}_wd_{args.lambdax}_split_{args.split}_loss_{args.loss}.png')

    # Log training loss
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    log_path = f'{output}/model_layers_{"_".join(map(str, args.layers))}_batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_act_{args.act}_wd_{args.lambdax}_split_{args.split}_loss_{args.loss}.log'
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    for epoch, loss in enumerate(losses):
        logger.info('Epoch %s: Loss %s', epoch, loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a neural network on 2D data')
    parser.add_argument('--folder', type=project_path, default='data/mlp', help='Folder containing the data files')
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 64, 64, 1], help='List specifying the number of neurons in each layer')
    parser.add_argument('--split', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--act', type=str, default='relu', help='Activation function for the hidden layers')
    parser.add_argument('--lambdax', type=float, default=0.0, help='Regularization parameter for the loss function')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use')
    parser.add_argument('--load', type=project_path, default=None, help='Path to a pretrained model to use for transfer learning')
    parser.add_argument('--scheduler', type=str, default='linear', help='Type of learning rate scheduler to use')
    parser.add_argument('--output', type=project_path, default=None, help='Folder to save the output files')
    args = parser.parse_args()

    main(args)
