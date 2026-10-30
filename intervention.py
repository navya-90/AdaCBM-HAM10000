"""Doctor-in-the-loop concept intervention, defined ONCE so the dashboard and the tests
use exactly the classification layer of the model being explained."""
import torch


def effective_head(model):
    """(W_eff, b_cls): the masked association matrix and class bias used in model.forward."""
    return model._get_weight_mat().detach(), model.asso_mat_bias.detach()


def logits_from_scores(model, scores):
    """logits = scores @ (W*M)^T + b_cls  -- identical to the last two lines of forward()."""
    W, b = effective_head(model)
    scores = torch.as_tensor(scores, dtype=W.dtype, device=W.device)
    return scores @ W.t() + b


def concept_classes(model):
    """Class index of every *selected* concept, taken from the saved concept metadata."""
    return model.concept2cls.view(-1).tolist()
