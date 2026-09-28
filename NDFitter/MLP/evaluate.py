import torch
from torch.autograd import Variable
from NDFitter.MLP.model import FeedForwardNN
from NDFitter.MLP.data.dataloader import get_dataloader
import logging
import argparse

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Define your model
    model = FeedForwardNN([2, 64, 64, 1])
    model.load_state_dict(torch.load("model_weights.pt"))

    # Get dataloaders
    _, test_loader = get_dataloader()

    # Evaluate the model
    test_loss = evaluate(model, test_loader)
    print(f'Test Loss: {test_loss:.4f}')

     # Log testing loss
    logger = logging.getLogger('TestingLogger')
    logger.setLevel(logging.INFO)
    log_path = f'{args.folder}/test_log_layers_{"_".join(map(str, args.layers))}_batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}.log'
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('Test Loss: %s', test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network on 2D data')
    args = parser.parse_args()

    main(args)
