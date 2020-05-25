import random
import torch
from torch.nn import MSELoss
from KD_Lib.common import BaseClass


def add_noise(x, variance=0.1):
    """
    Function for adding noise

    :param x (torch.FloatTensor): Input for adding noise
    :param variance (float): Variance for adding noise 
    """

    return x * (1 + (variance**0.5) * torch.randn_like(x))


class NoisyTeacher(BaseClass):
    """
    Implementation of Knowledge distillation using a noisy teacher from the paper "Deep 
    Model Compression: Distilling Knowledge from Noisy Teachers" https://arxiv.org/pdf/1610.09650.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (Loss function): Loss function to calculate loss between student and teacher predictions
    :param alpha (float): Threshold for deciding if noise needs to be added
    :param noise_variance (float): Variance parameter for adding noise
    :param loss (str): Loss used for training
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """
    
    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 optimizer_teacher, optimizer_student,
                 loss_fn=MSELoss(), alpha=0.5, noise_variance=0.1,
                 loss='MSE', temp=20.0, distil_weight=0.5, device='cpu', 
                 log=False, logdir='./Experiments'):
        super(NoisyTeacher, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss,
            temp,
            distil_weight,
            device,
            log,
            logdir
        )

        self.loss_fn = loss_fn
        self.alpha = alpha
        self.noise_variance = noise_variance

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model 
        :param y_true (torch.FloatTensor): Original label
        """
        
        if random.uniform(0, 1) <= self.alpha:
            y_pred_teacher = add_noise(y_pred_teacher, self.noise_variance)
        return self.loss_fn(y_pred_student, y_pred_teacher)
