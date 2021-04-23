import torch

from torch import nn


class Gen(nn.Module):
    def __init__(self, attr_num, attr_present_dim, hidden_dim, user_emb_dim):
        super(Gen, self).__init__()

        self.embed = nn.Embedding(2 * attr_num, attr_present_dim)
        self.linear = nn.Sequential(
            nn.Linear(attr_num * attr_present_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, user_emb_dim),
            nn.Tanh()
        )
        self.attr_num = attr_num
        self.attr_present_dim = attr_present_dim

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, attribute_id):
        attri_present = self.embed(attribute_id)
        attri_feature = attri_present.reshape(-1, self.attr_num * self.attr_present_dim)
        fake_user = self.linear(attri_feature)

        return fake_user


class Discriminator(nn.Module):
    def __init__(self, attr_num, attr_present_dim, hidden_dim, user_emb_dim):
        super(Discriminator, self).__init__()

        self.embed = nn.Embedding(2 * attr_num, attr_present_dim)
        self.linear = nn.Sequential(
            nn.Linear(attr_num * attr_present_dim + user_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, user_emb_dim),
            nn.Tanh()
        )
        self.attr_num = attr_num
        self.attr_present_dim = attr_present_dim
        self.sig = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, attribute_id, user_emb):
        attri_present = self.embed(attribute_id)
        attri_feature = attri_present.reshape(-1, self.attr_num * self.attr_present_dim)
        emb = torch.cat((attri_feature, user_emb), 1)

        D_logit = self.linear(emb)
        D_prob = self.sig(D_logit)

        return D_logit


if __name__ == '__main__':
    attr_num = 18
    attr_present_dim = 5
    hidden_dim = 100
    user_emb_dim = attr_num

    batch_size = 2

    attribute_id = torch.randint(low=0, high=10, size=(batch_size, attr_num))

    user_emb = torch.randn((batch_size, user_emb_dim))

    gen1 = Gen(attr_num, attr_present_dim, hidden_dim, user_emb_dim)
    res1 = gen1(attribute_id)

    dis1 = Discriminator(attr_num, attr_present_dim, hidden_dim, user_emb_dim)
    res2 = dis1(attribute_id, user_emb)

    print()



















