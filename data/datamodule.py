from typing import Optional
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule
from data.dataset import captcha_dataset
from torch.utils.data import DataLoader


class captcha_dm(LightningDataModule):
    def __init__(self, batch_size=128, num_workers=8):
        super(captcha_dm, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str]) -> None:
        self.train_dataset = captcha_dataset('train')
        self.val_dataset = captcha_dataset('val')
        self.test_dataset = captcha_dataset('test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=self.num_workers != 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=self.num_workers != 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=self.num_workers != 0)


if __name__ == '__main__':
    dm = captcha_dm()
    dm.setup(stage=None)
    it = dm.train_dataloader()
    print(next(iter(it)))
