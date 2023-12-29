import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from KD_Lib.KD.common.utils import sharpness, sharpness_gap, AverageMeter

class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        scheduler_teacher,
        scheduler_student,
        loss_ce,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.scheduler_teacher = scheduler_teacher
        self.scheduler_student = scheduler_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)
            layout = {"Metrics": {"Loss":     ["Multiline", ["Loss Teacher", "Loss Student", "CE Student", "KD Student"]],
                                  "Accuracy": ["Multiline", ["Accuracy Teacher Train", "Accuracy Student Train", 
                                                             "Accuracy Student Val", "Accuracy Teacher Val"]],
                                  "Sharpness": ["Multiline", ["Sharpness Teacher Train", "Sharpness Student Train",
                                                              "Sharpness Student Val", "Sharpness Teacher Val"]],
                                  "Sharpness Gap": ["Multiline", ["Sharpness Gap Train", "Sharpness Gap Val"]]}}
            self.writer.add_custom_scalars(layout)

        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_ce = loss_ce.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def train_teacher(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher_kd.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.teacher_model.train()
        length_of_dataset = len(self.train_loader.dataset)
        epoch_loss, train_sharp = AverageMeter(), AverageMeter()
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Teacher... ")

        for ep in range(epochs):
            epoch_loss.reset()
            correct = 0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)
                if isinstance(out, tuple):
                    out = out[0]
                train_sharp.update(sharpness(out))

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.ce_fn(out, label)
                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss.update(loss.item())

            epoch_acc = correct / length_of_dataset
            epoch_val_acc, val_sharp = self.evaluate(teacher=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

            if self.log:
                self.writer.add_scalar("Loss Teacher", epoch_loss.avg, ep)
                self.writer.add_scalar("Accuracy Teacher Train", epoch_acc, ep)
                self.writer.add_scalar("Accuracy Teacher Val", epoch_val_acc, ep)
                self.writer.add_scalar("Sharpness Teacher Train", train_sharp.avg, ep)
                self.writer.add_scalar("Sharpness Teacher Val", val_sharp, ep)

            print(f"[{ep+1}] LR: {self.scheduler_teacher.get_last_lr()[0]:.1e}, Loss: {(epoch_loss.avg):.4f},", 
                  f"Acc: {epoch_acc:.4f}, ValAcc: {epoch_val_acc:.4f}")
            print("-" * 70)

            self.scheduler_teacher.step()

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)

    def _train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student_kd.pt",
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        epoch_loss, epoch_ce_loss, epoch_kd_loss = AverageMeter(), AverageMeter(), AverageMeter()
        g_sharp_train, t_sharp_train, s_sharp_train = AverageMeter(), AverageMeter(), AverageMeter()
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")
        for ep in range(epochs):
            epoch_loss.reset(), epoch_ce_loss.reset(), epoch_kd_loss.reset()
            correct = 0

            self.teacher_model.eval()
            self.student_model.train()

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)
                
                self.optimizer_student.zero_grad()

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)
                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                g_sharp, t_sharp, s_sharp = sharpness_gap(teacher_out, student_out)
                g_sharp_train.update(g_sharp)
                t_sharp_train.update(t_sharp)
                s_sharp_train.update(s_sharp)
                loss, ce_loss, kd_loss = self.calculate_kd_loss(student_out, teacher_out, label)

                loss.backward()
                self.optimizer_student.step()

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                epoch_loss.update(loss.item())
                epoch_ce_loss.update(ce_loss.item())
                epoch_kd_loss.update(kd_loss.item())

            epoch_acc = correct / length_of_dataset

            _, epoch_val_acc, s_sharp_val = self._evaluate_model(self.student_model, verbose=True)
            _, _, t_sharp_val = self._evaluate_model(self.teacher_model, verbose=True)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Loss Student", epoch_loss.avg, ep)
                self.writer.add_scalar("CE Student", epoch_ce_loss.avg, ep)
                self.writer.add_scalar("KD Student", epoch_kd_loss.avg, ep)
                self.writer.add_scalar("Accuracy Student Train", epoch_acc, ep)
                self.writer.add_scalar("Accuracy Student Val", epoch_val_acc, ep)
                self.writer.add_scalar("Sharpness Student Train", s_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Student Val", s_sharp_val, ep)
                self.writer.add_scalar("Sharpness Teacher Train", t_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Gap Train", g_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Teacher Val", t_sharp_val, ep)
                self.writer.add_scalar("Sharpness Gap Val", t_sharp_val - s_sharp_val, ep)

            print(
                f"[{ep+1}] LR: {self.scheduler_student.get_last_lr()[0]:.1e},",
                f"Loss: {(epoch_loss.avg):.4f}, CE: {(epoch_ce_loss.avg):.4f}, KD: {(epoch_kd_loss.avg):.4f},", 
                f"Acc: {epoch_acc:.4f}, ValAcc: {epoch_val_acc:.4f}")
            print("-" * 80)
            self.scheduler_student.step()

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student_kd.pt",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self._train_student(epochs, plot_losses, save_model, save_model_pth)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []
        sharp = AverageMeter()

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                sharp.update(sharpness(output))
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / length_of_dataset
        return outputs, accuracy, sharp.avg

    def evaluate(self, teacher=False, verbose=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy, sharp = self._evaluate_model(model)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
        return accuracy, sharp

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 60)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
