# imports
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_factory import ModelFactory

from config_recvis import opts
from torch.optim import lr_scheduler


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--lr_step",
        type=int,
        default=10,
        metavar="LS",
        help="step size for learning rate scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="G",
        help="factor for learning rate scheduler",
    )
    args = parser.parse_args()
    return args


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
    train_accuracies: list,
    train_losses: list
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
        accuracies: accuracies list
    """
    model.train()
    correct = 0
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        train_loss += loss.data.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    train_accuracy= 100.0 * correct / len(train_loader.dataset)
    train_accuracies.append(train_accuracy)
    train_loss/= len(train_loader.dataset)
    train_losses.append(train_loss)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            train_accuracy,
        )
    )


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    validation_accuracy= 100.0 * correct / len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            validation_accuracy,
        )
    )
    return validation_loss, validation_accuracy


def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms = ModelFactory(args.model_name).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler= lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    # Loop over the epochs
    best_val_loss = 1e8
    train_accuracies= []
    val_accuracies=[]

    train_losses= []
    val_losses=[]

    count=0
    # freeze the 
    model.freeze()
    for epoch in range(0, args.epoch + 1):
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args, train_accuracies, train_losses)
        # validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda, epoch)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        # update learning rate
        scheduler.step()
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )

    ## now unfreeze the model parameters
    model.unfreeze()
    model.load_state_dict(torch.load('experiment/model_best.pth'))
    print('Now unfreeze model')
    for epoch in range(0, 40 + 1):
        # training loop
        train(model, optimizer, train_loader, use_cuda, epoch, args, train_accuracies, train_losses)
        # validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda, epoch)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        # update learning rate
        scheduler.step()
        model_file = args.experiment + "/model_" + str(epoch)+ "_"+ str(val_accuracy)+ ".pth"
        torch.save(model.state_dict(), model_file)
        if val_loss > best_val_loss:
          count+=1
          if count> 5:
            print('stopping the train, restart')
            break
        else:
          count=0
          best_val_loss = val_loss
          best_model_file = args.experimen + "/model_best.pth"
          torch.save(model.state_dict(), best_model_file)



    # model.load_state_dict(torch.load('experiment/model_best.pth'))
    # # model.freeze()
    # for epoch in range(0, 15 + 1):
    #     # training loop
    #     train(model, optimizer, train_loader, use_cuda, epoch, args, train_accuracies, train_losses)
    #     # validation loop
    #     val_loss, val_accuracy = validation(model, val_loader, use_cuda, epoch)
    #     val_accuracies.append(val_accuracy)
    #     val_losses.append(val_loss)
    #     # update learning rate
    #     scheduler.step()
    #     if val_loss < best_val_loss:
    #         # save the best model for validation
    #         best_val_loss = val_loss
    #         best_model_file = "experiment" + "/model_best.pth"
    #         torch.save(model.state_dict(), best_model_file)
    #     # also save the model every epoch
    #     model_file = "experiment" + "/model_" + str(epoch) + ".pth"
    #     torch.save(model.state_dict(), model_file)
    #     print(
    #         "Saved model to "
    #         + model_file
    #         + f". You can run `python evaluate.py --model_name {args.model_name} --model "
    #         + best_model_file
    #         + "` to generate the Kaggle formatted csv file\n"
    #     )

    f, ax = plt.subplots(1, 2 ,figsize=(20,7))
    ax[0].plot(range(len(val_accuracies)) , val_accuracies , label= 'Val',marker='.')
    ax[0].plot(range(len(train_accuracies)) , train_accuracies , label='Train',marker='.')
    ax[0].set_title('The accuracy')
    ax[0].legend()
    ax[0].set_xlabel('epoch')

    ax[1].plot(range(len(val_losses)) , val_losses , label= 'Val',marker='.')
    ax[1].plot(range(len(train_losses)) , train_losses , label='Train',marker='.')
    ax[1].legend()
    ax[1].set_title('Loss')
    ax[1].set_xlabel('epoch')

    plt.show()




if __name__ == "__main__":
    main()
