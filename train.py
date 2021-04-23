import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from net import Gen, Discriminator
from dataset import MyDataset

if __name__ == '__main__':
    # Hyper parameters
    alpha = 0
    attr_num = 18  # the number of attribute
    attr_present_dim = 5  # the dimension of attribute present
    batch_size = 128
    hidden_dim = 100  # G hidden layer dimension
    user_emb_dim = attr_num

    iters = 100
    D_range = 1
    G_range = 1
    device = 'cpu'


    # model
    generator = Gen(attr_num, attr_present_dim, hidden_dim, user_emb_dim)
    generator.init_weights()
    generator.to(device)

    discriminator = Discriminator(attr_num, attr_present_dim, hidden_dim, user_emb_dim)
    discriminator.init_weights()
    discriminator.to(device)

    # optimizer
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0001, weight_decay=alpha)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0001, weight_decay=alpha)

    # dataLoader
    train_dataLoader = DataLoader(MyDataset('train'),
                                  batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


    # train
    for iter in range(iters):
        for D_iter in range(D_range):
            print('DDD')

            for i, batch in enumerate(train_dataLoader):
                user, item, attr, real_user_emb, user2, item2, attr2, neg_user_emb = batch[0], batch[1], batch[2], batch[3], \
                                                                                     batch[4], batch[5], batch[6], batch[7]
                attr = attr.long()
                attr2 = attr2.long()
                real_user_emb = real_user_emb.float()
                neg_user_emb = neg_user_emb.float()

                fake_user_emb = generator(attr)
                D_logit_real = discriminator(attr, real_user_emb)
                D_logit_fake = discriminator(attr, fake_user_emb)
                D_logit_counter = discriminator(attr2, neg_user_emb)

                real_labels = torch.ones_like(D_logit_real)
                fake_labels = torch.zeros_like(D_logit_fake)
                labels = torch.zeros_like(D_logit_counter)

                D_loss_real = torch.mean(
                    -real_labels * torch.log(torch.sigmoid(D_logit_real)) - (1 - real_labels) * torch.log(1 - torch.sigmoid(D_logit_real)))
                D_loss_fake = torch.mean(
                    -fake_labels * torch.log(torch.sigmoid(D_logit_fake)) - (1 - fake_labels) * torch.log(1-torch.sigmoid(D_logit_fake)))
                D_loss_counter = torch.mean(
                    -labels * torch.log(torch.sigmoid(D_logit_counter)) - (1 - labels) * torch.log(1 - torch.sigmoid(D_logit_counter))
                )

                D_loss = (1 - alpha) * (D_loss_real + D_loss_fake + D_loss_counter)

                optimizerD.zero_grad()
                D_loss.backward()
                optimizerD.step()

                print('iter:{}, D_loss:{}'.format(iter, D_loss.item()))


        for G_iter in range(G_range):
            print('GGG')

            for i, batch in enumerate(train_dataLoader):
                user, item, attr, real_user_emb = batch[0], batch[1], batch[2], batch[3]

                attr = attr.long()
                real_user_emb = real_user_emb.float()
                fake_user_emb = generator(attr)
                D_logit_fake = discriminator(attr, fake_user_emb)

                fake_labels = torch.ones_like(D_logit_fake)
                G_loss = (1 - alpha) * torch.mean(
                    -fake_labels * torch.log(torch.sigmoid(D_logit_fake)) - (1 - fake_labels) * torch.log(1-torch.sigmoid(D_logit_fake)))

                optimizerG.zero_grad()
                G_loss.backward()
                optimizerG.step()

                print('iter:{}, G_loss:{}'.format(iter, G_loss.item()))





