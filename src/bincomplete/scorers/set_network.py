"""Transformer bin-completion queries and membership-training objectives."""

import torch
from torch import nn
from torch.nn import functional as F


class BinCompletion(nn.Module):
    def __init__(self, embedding_dim, width=256, queries=4, heads=4, layers=2, feedforward=512):
        super().__init__()
        self.input = nn.Linear(embedding_dim + 1, width)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.output = nn.Linear(width, embedding_dim)

    def forward(self, core, mask, log_coverage):
        x = self.input(torch.cat([core, log_coverage.unsqueeze(-1)], dim=-1))
        x = self.encoder(x, src_key_padding_mask=~mask)
        q = self.queries.unsqueeze(0).expand(len(core), -1, -1)
        q, _ = self.attention(q, x, x, key_padding_mask=~mask, need_weights=False)
        return F.normalize(self.output(q), dim=-1)


def set_loss(queries, candidates, labels, candidate_mask, objective, temperature=0.1):
    if objective not in ["compatibility", "completion", "matched_completion"]:
        raise ValueError("Unknown set objective")
    s = torch.einsum("bqd,bnd->bqn", queries, candidates) / temperature
    if objective == "matched_completion":
        from scipy.optimize import linear_sum_assignment

        episodes = []
        for b in range(len(s)):
            valid = candidate_mask[b]
            positive = torch.nonzero(labels[b] & valid).flatten()
            if len(positive) > queries.shape[1]:
                raise ValueError("More hidden targets than predicted query slots")
            if not len(positive):
                episodes.append(F.softplus(s[b][:, valid]).mean())
                continue
            assignment = -s[b][:, positive].detach().cpu().numpy()
            qi, pi = linear_sum_assignment(assignment)
            qi = torch.as_tensor(qi, device=s.device)
            targets = positive[torch.as_tensor(pi, device=s.device)]
            logits = s[b, qi].masked_fill(~valid[None, :], -1e4)
            episodes.append((torch.logsumexp(logits, 1) - s[b, qi, targets]).mean())
        return torch.stack(episodes).mean()
    if objective == "compatibility":
        loss = F.binary_cross_entropy_with_logits(s.amax(1), labels.float(), reduction="none")
        p, n = labels & candidate_mask, ~labels & candidate_mask
        return ((loss * p).sum(1) / p.sum(1).clamp_min(1) + (loss * n).sum(1) / n.sum(1).clamp_min(1)).mean()
    present = (labels & candidate_mask).any(1)
    per_episode = s.sum((1, 2)) * 0
    if (~present).any():
        loss = F.softplus(s[~present].amax(1))
        per_episode[~present] = (loss * candidate_mask[~present]).sum(1) / candidate_mask[~present].sum(
            1
        ).clamp_min(1)
    if not present.any():
        return per_episode.mean()
    s, labels, candidate_mask = s[present], labels[present], candidate_mask[present]
    s = s.masked_fill(~candidate_mask[:, None, :], -1e4)
    positive = s.masked_fill(~labels[:, None, :], -1e4)
    query_loss = (torch.logsumexp(s, 2) - torch.logsumexp(positive, 2)).mean(1)
    missing_loss = -(s.amax(1).log_softmax(1) * labels.float()).sum(1) / labels.sum(1).clamp_min(1)
    per_episode[present] = query_loss + missing_loss
    return per_episode.mean()
