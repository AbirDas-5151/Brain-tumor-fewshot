import torch
import torch.nn as nn
import torch.nn.functional as F


from collections import defaultdict

def create_few_shot_task(dataset, n_way=3, k_shot=5, q_query=5):
    class_to_indices = defaultdict(list)

    for i, (_, label) in enumerate(dataset):
        class_to_indices[label].append(i)

    selected_classes = random.sample(class_to_indices.keys(), n_way)

    support_x, support_y, query_x, query_y = [], [], [], []

    for class_idx, cls in enumerate(selected_classes):
        indices = class_to_indices[cls]
        sampled = random.sample(indices, k_shot + q_query)
        support = sampled[:k_shot]
        query = sampled[k_shot:]

        support_x.extend([dataset[i][0] for i in support])
        support_y.extend([class_idx] * k_shot)

        query_x.extend([dataset[i][0] for i in query])
        query_y.extend([class_idx] * q_query)

    support_x = torch.stack(support_x)
    query_x = torch.stack(query_x)
    return support_x, torch.tensor(support_y), query_x, torch.tensor(query_y)


import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder  # Typically a CNN backbone like resnet18

    def forward(self, support, support_labels, query):
        n_classes = len(torch.unique(support_labels))
        embeddings = self.encoder(torch.cat([support, query], dim=0))
        support_embeddings = embeddings[:len(support)]
        query_embeddings = embeddings[len(support):]

        prototypes = []
        for c in range(n_classes):
            class_embeddings = support_embeddings[support_labels == c]
            proto = class_embeddings.mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes)
        dists = torch.cdist(query_embeddings, prototypes)
        return -dists  # Negative distance as similarity


class MatchingNet(nn.Module):
    def __init__(self, encoder):
        super(MatchingNet, self).__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query):
        support_embeddings = self.encoder(support)
        query_embeddings = self.encoder(query)

        # Cosine similarity
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        sims = torch.matmul(query_embeddings, support_embeddings.T)  # [Q, S]
        softmax_weights = F.softmax(sims, dim=1)

        one_hot = F.one_hot(support_labels, num_classes=len(torch.unique(support_labels))).float()
        preds = torch.matmul(softmax_weights, one_hot)  # [Q, C]
        return preds

from torch import autograd

class MAML(nn.Module):
    def __init__(self, encoder, lr_inner=0.01):
        super(MAML, self).__init__()
        self.encoder = encoder
        self.lr_inner = lr_inner
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, support, support_labels):
        fast_weights = list(self.encoder.parameters())
        support_preds = self.encoder(support)
        loss = self.loss_fn(support_preds, support_labels)

        grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
        adapted_weights = [w - self.lr_inner * g for w, g in zip(fast_weights, grads)]
        return adapted_weights

    def forward(self, support, support_labels, query):
        adapted_weights = self.adapt(support, support_labels)
        for p, w in zip(self.encoder.parameters(), adapted_weights):
            p.data = w.data
        return self.encoder(query)
