import numpy as np

def sharpness(logits, eps=1e-9):
    """Computes the sharpness of the logits.
    Args:
        logits: Tensor of shape [batch_size, num_classes] containing the logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness of the logits.
    """
    logits = logits.detach().cpu().numpy()
    return np.log(np.exp(logits).sum(axis=1) + eps)

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
    return np.mean(teacher_sharpness - student_sharpness), np.mean(teacher_sharpness), np.mean(student_sharpness)

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