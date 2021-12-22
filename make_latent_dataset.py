import pickle
from tqdm import tqdm
import lmdb
import torch
from torch.utils.data import DataLoader
from vqvae import VQVAE
import torchvision
import torchvision.transforms as transforms
from collections import namedtuple

Data = namedtuple('Data', ['top', 'bottom'])


def extract(lmdb_env, dataloader, model, device):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        dataloader = tqdm(dataloader)

        for img, label in dataloader:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for top, bottom in zip(id_t, id_b):
                data = Data(top=top, bottom=bottom)
                txn.put(str(index).encode('utf-8'), pickle.dumps(data))
                index += 1
                dataloader.set_description(f"Inserted: {index}")

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False)

    model = VQVAE.load_from_checkpoint("9epochs.ckpt")
    model.to(device)
    model.eval()

    env = lmdb.open("latent_code_data", map_size=100 * 1024 * 1024 * 1024)
    extract(env, train_loader, model, device)
    print("Success")
