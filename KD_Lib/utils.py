import os, json, pprint, typing, random
import numpy as np
import optuna
import torch

# Get Optimizer and Scheduler
def get_optim_sched(models: list, cfg, single=False):
    os = {'optims': [], 'scheds': []}
        
    if single:
        params = torch.nn.ModuleList(models).parameters()
        opt, sched = get_single_opt_sched(params, cfg)
        os['optims'] = opt
        os['scheds'] = sched
        return os
    
    for model in models:
        params = list(model.parameters())
        opt, sched = get_single_opt_sched(model.parameters(), cfg)
        os['optims'].append(opt)
        os['scheds'].append(sched)
    return os

def get_single_opt_sched(params, cfg):
    opt = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WD)
    if cfg.SCHEDULER == 'cos':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
    elif cfg.SCHEDULER == 'step':
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'lin':
        sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=cfg.EPOCHS, start_factor=1, end_factor=cfg.LR_MIN/cfg.LR)
    return opt, sched


# Utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, file):
        self.file = file
        self.pp = pprint.PrettyPrinter(depth=2)
        self.save_log(f"Logging to {file}.")
        
    def save_log(self, text):
        if type(text) is dict:
            text = self.pp.pformat(text)
        print(text)
        with open(self.file, 'a') as f:
            f.write(text + '\n')

class MultiPruner(optuna.pruners.BasePruner):
    def __init__(self, pruners: typing.Iterable[optuna.pruners.BasePruner], pruning_condition: str = "any") -> None:
        self._pruners = tuple(pruners)
        self._pruning_condition_check_fn = None
        if pruning_condition == "any":
            self._pruning_condition_check_fn = any
        elif pruning_condition == "all":
            self._pruning_condition_check_fn = all
        else:
            raise ValueError(f"Invalid pruning ({pruning_condition}) condition passed!")
        assert self._pruning_condition_check_fn is not None

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
         return self._pruning_condition_check_fn(pruner.prune(study, trial) for pruner in self._pruners)

def log_cfg(cfg):
    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)
    with open(os.path.join(cfg.LOG_DIR, f"{cfg.EXP}.json"), "w") as file:
        json.dump(cfg.__dict__, file)

def set_environment(seed, device):
    if int(device) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Sharpness Metric
def sharpness(logits, eps=1e-9, clip=70):
    """Computes the sharpness of the logits.
    Args:
        logits: Tensor of shape [batch_size, num_classes] containing the logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness of the logits.
    """
    logits = logits.detach().cpu().numpy()
    if clip != np.inf:
        logits = np.clip(logits, -clip, clip)
    else: 
        logits = logits.astype(np.float128)
    return np.mean(np.log(np.exp(logits).sum(axis=1) + eps))

def sharpness_gap(teacher_logits, student_logits, eps=1e-9):
    """Computes the sharpness gap between the teacher and student logits.
    Args:
        teacher_logits: Tensor of shape [batch_size, num_classes] containing the teacher logits.
        student_logits: Tensor of shape [batch_size, num_classes] containing the student logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness gap between the teacher and student logits.
    """
    teacher_sharpness = sharpness(teacher_logits, eps)
    student_sharpness = sharpness(student_logits, eps)
    return teacher_sharpness - student_sharpness, teacher_sharpness, student_sharpness


# CKA Metric ("Similarity of Neural Network Representations Revisited")
def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)

def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)