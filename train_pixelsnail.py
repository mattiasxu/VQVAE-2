import torch
import pytorch_lightning as pl
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import lmdb
from pixelsnail import HierarchicalPixelSNAIL
from make_latent_dataset import Data


class LatentDataset(Dataset):
    def __init__(self, path, n_embed=512):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.n_embed = n_embed

        if not self.env:
            raise IOError('Cannot open dataset', path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            data = pickle.loads(txn.get(key))
        top = one_hot(torch.from_numpy(data.top), self.n_embed).permute(2, 0, 1).type(torch.FloatTensor)
        bot = one_hot(torch.from_numpy(data.bottom), self.n_embed).permute(2, 0, 1).type(torch.FloatTensor)
        return top, bot


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = LatentDataset("latent_code_data", n_embed=512)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

    model = HierarchicalPixelSNAIL(512, 128, 10, 3, 5, 16, 128)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, loader)
