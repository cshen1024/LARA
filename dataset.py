import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        if mode == 'train':
            self.files1 = np.array(pd.read_csv('train_data.csv', header=None))
            self.files2 = np.array(pd.read_csv('neg_data.csv', header=None))
        # if mode == 'test':
        #     self.test_item = np.array(pd.read_csv('test_item.csv',
        #                                           header=None).astype(np.int32))
        #     self.test_attribute = np.array(pd.read_csv('test_attribute.csv',
        #                                                header=None).astype(np.int32))

        self.user_emb_matrix = np.array(pd.read_csv('user_emb.csv', header=None))

    def __getitem__(self, index):
        data = self.files1[index]

        user = np.array([data[0]])
        item = np.array([data[1]])
        attr = data[2][1: -1].split()
        for i in range(len(attr)):
            attr[i] = int(attr[i])
        attr = np.array(attr)

        real_user_emb = self.user_emb_matrix[user]

        data2 = self.files2[index]
        user2 = np.array([data2[0]])
        item2 = np.array([data2[1]])
        attr2 = data2[2][1: -1].split()
        for i in range(len(attr2)):
            attr2[i] = int(attr2[i])
        attr2 = np.array(attr2)

        neg_user_emb = self.user_emb_matrix[user2]

        return user, item, attr, real_user_emb.squeeze(), user2, item2, attr2, neg_user_emb.squeeze()

    def __len__(self):
        return min(len(self.files1), len(self.files2))


if __name__ == '__main__':
    train_dataLoader = DataLoader(MyDataset('train'),
                                  batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    for i, batch in enumerate(train_dataLoader):
        user, item, attr, real_user_emb, user2, item2, attr2, neg_user_emb = batch[0], batch[1], batch[2], batch[3], \
                                                                             batch[4], batch[5], batch[6], batch[7]
