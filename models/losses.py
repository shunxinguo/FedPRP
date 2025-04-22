import torch
import torch.nn as nn


class DROLoss(nn.Module):
    def __init__(self, temperature=1, base_temperature=1, class_weights=None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = torch.stack(torch.tensor(class_weights))
    def count(self, batch_feats, batch_targets, centroid_feats, centroid_targets,learnable_epsilons):

        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets)
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])
        pairwise = -1 * self.pairwise_cosine_sim(train_prototypes, batch_feats)

        # epsilons
        if learnable_epsilons is not None:
            learnable_epsilons.to(self.args.device)
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(self.args.device)
            a = pairwise.clone()
            b = learnable_epsilons
            c = b.to(self.args.device)
            pairwise[mask] = a[mask] - c[batch_targets]

        logits = torch.div(pairwise, self.temperature)

        # compute log_prob
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)

        # compute mean of log-likelihood over positive
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(self.args.device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        loss = loss.sum() / len(classes)


        return loss
