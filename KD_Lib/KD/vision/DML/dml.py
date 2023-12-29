import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class DML:
    """
    Implementation of "Deep Mutual Learning" https://arxiv.org/abs/1706.00384

    :param student_cohort (list/tuple): Collection of student models
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param student_optimizers (list/tuple): Collection of Pytorch optimizers for training students
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        students,
        train_loader,
        val_loader,
        optimizers,
        schedulers,
        loss_ce,
        loss_kd,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        
        self.students = students
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.loss_ce = loss_ce
        self.loss_kd = loss_kd
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)
            layout = {"Metrics": {"Loss":     ["Multiline", ["Loss Teacher", "Loss Student", "CE Student", "KD Student"]],
                                  "Accuracy": ["Multiline", ["Accuracy Teacher Train", "Accuracy Student Train", 
                                                             "Accuracy Student Val", "Accuracy Teacher Val"]]}}
            self.writer.add_custom_scalars(layout)

        try:
            torch.Tensor(0).to(device)
            self.device = device
        except:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = "cpu"

    def train_students(
        self,
        epochs=20,
        save_model=True,
        save_model_path="./models/student_dml.pt",
    ):

        for student in self.students:
            student.to(self.device)
            student.train()

        num_students = len(self.students)
        data_len = len(self.train_loader)
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.students[0].state_dict())
        self.best_student = self.students[-1]

        print("\nTraining Teacher and Student...")

        for ep in range(epochs):
            epoch_loss, epoch_ce_loss, epoch_kd_loss = 0.0, 0.0, 0.0
            t_correct, s_correct = 0, 0

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.optimizers:
                    optim.zero_grad()

                for i in range(num_students):
                    student_loss = 0
                    for j in range(num_students):
                        if i == j:
                            continue
                        loss_kd = self.loss_kd(F.log_softmax(self.students[i](data), dim=1), 
                                               F.log_softmax(self.students[j](data), dim=1))
                        student_loss += loss_kd
                    student_loss /= num_students - 1
                    loss_ce = self.loss_ce(self.students[i](data), label)
                    student_loss += loss_ce
                    student_loss.backward()
                    self.optimizers[i].step()

                epoch_loss += student_loss.item()
                epoch_ce_loss += loss_ce.item()
                epoch_kd_loss += loss_kd.item()

                predictions = []
                correct_preds = []
                for i, student in enumerate(self.students):
                    predictions.append(student(data).argmax(dim=1, keepdim=True))
                    correct_preds.append(predictions[i].eq(label.view_as(predictions[i])).sum().item())

                t_correct += correct_preds[0]
                s_correct += correct_preds[-1]

            t_epoch_acc = t_correct / length_of_dataset
            s_epoch_acc = s_correct / length_of_dataset
            
            val_accs = self.evaluate(verbose=False)
            if val_accs[-1] > best_acc:
                best_acc = val_accs[-1]
                self.best_student_model_weights = deepcopy(student.state_dict())
                self.best_student = student

            if self.log:
                self.writer.add_scalar("Loss Student", epoch_loss/data_len, ep)
                self.writer.add_scalar("CE Student", epoch_ce_loss/data_len, ep)
                self.writer.add_scalar("KD Student", epoch_kd_loss/data_len, ep)
                self.writer.add_scalar("Accuracy Teacher Train", t_epoch_acc, ep)
                self.writer.add_scalar("Accuracy Student Train", s_epoch_acc, ep)
                self.writer.add_scalar("Accuracy Teacher Val", val_accs[0], ep)
                self.writer.add_scalar("Accuracy Student Val", val_accs[-1], ep)

            print(f"[{ep+1}] LR: {self.schedulers[0].get_last_lr()[0]:.1e},",
                  f"Loss: {(epoch_loss/data_len):.4f}, CE: {epoch_ce_loss/data_len:.4f}, KD: {epoch_kd_loss/data_len:.4f}\n",
                  f"[T] Acc: {t_epoch_acc:.4f}, ValAcc: {val_accs[0]:.4f}, [S] Acc: {s_epoch_acc:.4f}, ValAcc: {val_accs[-1]:.4f}")
            print("-" * 100)
            for sched in self.schedulers:
                sched.step()

        self.best_student.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.best_student.state_dict(), save_model_path)

    def _evaluate_model(self, model, verbose=False, name=""):
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

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        if verbose:
            print(f"M{name} Accuracy: {correct/length_of_dataset}")

        epoch_val_acc = correct / length_of_dataset
        return outputs, epoch_val_acc

    def evaluate(self, verbose=False):
        """
        Evaluate method for printing accuracies of the trained student networks

        """
        val_accs = []
        for i, student in enumerate(self.students):
            model = deepcopy(student).to(self.device)
            _, val_acc_i = self._evaluate_model(model, name=i)
            val_accs.append(val_acc_i)
        if verbose:
            print(f"Teacher Accuracy: {val_accs[0]:.4f}, Student Accuracy: {val_accs[1]:.4f}")
        return val_accs

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        print("-" * 80)
        for i, student in enumerate(self.students):
            student_params = sum(p.numel() for p in student.parameters())
            print(f"Total parameters for the student network {i} are: {student_params}")