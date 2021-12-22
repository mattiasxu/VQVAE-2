import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vqvae import VQVAE
import pytorch_lightning as pl


def main():
    args = dict(
        batch_size=32,
        lr=0.000056,
        epochs=10
    )

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = VQVAE(
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=64,
        embed_dim=64,
        n_embed=512,
        params=config
    )

    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs)
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    main()
