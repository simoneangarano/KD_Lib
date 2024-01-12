import os, time
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from KD_Lib.utils import sharpness, sharpness_gap, AverageMeter, log_cfg

class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework
    """

    def __init__(
        self,
        models,
        loaders,
        optimizers,
        schedulers,
        losses,
        cfg
    ):
        
        self.train_loader, self.val_loader = loaders
        self.teacher_model, self.student_model = models
        self.optimizer_teacher, self.optimizer_student = optimizers
        self.scheduler_teacher, self.scheduler_student = schedulers
        self.loss_ce, self.loss_kd = losses
        self.cfg = cfg
        self.device = torch.device(self.cfg.DEVICE)

        if self.cfg.LOG:
            self.writer = SummaryWriter(self.cfg.TB_DIR)
            layout = {"Metrics": {"Loss":     ["Multiline", ["Loss Teacher", "Loss Student", "CE Student", "KD Student"]],
                                  "Accuracy": ["Multiline", ["Accuracy Teacher Train", "Accuracy Student Train", 
                                                             "Accuracy Student Val", "Accuracy Teacher Val"]],
                                  "Sharpness": ["Multiline", ["Sharpness Teacher Train", "Sharpness Student Train",
                                                              "Sharpness Student Val", "Sharpness Teacher Val"]],
                                  "Sharpness Gap": ["Multiline", ["Sharpness Gap Train", "Sharpness Gap Val"]]}}
            self.writer.add_custom_scalars(layout)

    def train_teacher(self, save_model=True):
        """
        Function that will be training the teacher
        """
        length_of_dataset = len(self.train_loader.dataset)
        epoch_loss, train_sharp = AverageMeter(), AverageMeter()
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        save_dir = os.path.dirname(self.cfg.TEACHER_WEIGHTS)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Teacher... ")

        for ep in range(self.cfg.EPOCHS):
            t0 = time.time()
            epoch_loss.reset()
            correct = 0
            self.set_models(mode='train_teacher')

            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)
                train_sharp.update(sharpness(out))

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.loss_ce(out, label)
                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss.update(loss.item())

            epoch_acc = correct / length_of_dataset
            epoch_val_acc, val_sharp = self.evaluate(teacher=True)
            self.cfg.VACC['T_LAST'] = epoch_val_acc

            if epoch_val_acc > self.cfg.VACC['T_BEST']:
                self.cfg.VACC['T_BEST'] = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

            if self.cfg.LOG:
                self.writer.add_scalar("Loss Teacher", epoch_loss.avg, ep)
                self.writer.add_scalar("Accuracy Teacher Train", epoch_acc, ep)
                self.writer.add_scalar("Accuracy Teacher Val", epoch_val_acc, ep)
                self.writer.add_scalar("Sharpness Teacher Train", train_sharp.avg, ep)
                self.writer.add_scalar("Sharpness Teacher Val", val_sharp, ep)
                log_cfg(self.cfg)

            print(f"[{ep+1}: {(time.time() - t0)/60.0:.1f}m] LR: {self.scheduler_teacher.get_last_lr()[0]:.1e}, Loss: {(epoch_loss.avg):.4f},", 
                  f"Acc: {epoch_acc:.4f}, ValAcc: {epoch_val_acc:.4f}")
            print("-" * 100)

            self.scheduler_teacher.step()

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), self.cfg.TEACHER_WEIGHTS)

    def _train_student(self, save_model=True):
        """
        Function to train student model - for internal use only.
        """
        length_of_dataset = len(self.train_loader.dataset)
        epoch_loss, epoch_ce_loss, epoch_kd_loss = AverageMeter(), AverageMeter(), AverageMeter()
        g_sharp_train, t_sharp_train, s_sharp_train = AverageMeter(), AverageMeter(), AverageMeter()
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(self.cfg.SAVE_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")
        for ep in range(self.cfg.EPOCHS):
            t0 = time.time()
            epoch_loss.reset(), epoch_ce_loss.reset(), epoch_kd_loss.reset()
            correct = 0
            self.set_models(mode='train_student')

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)
                
                self.optimizer_student.zero_grad()

                with torch.no_grad():
                    teacher_out = self.teacher_model(data, norm_feats=self.cfg.FEAT_NORM)
                student_out = self.student_model(data, norm_feats=self.cfg.FEAT_NORM)
                loss, ce_loss, kd_loss = self.calculate_kd_loss(student_out, teacher_out, label)

                if isinstance(teacher_out, tuple):
                    teacher_out = teacher_out[0]
                    student_out = student_out[0]
                g_sharp, t_sharp, s_sharp = sharpness_gap(teacher_out, student_out)
                g_sharp_train.update(g_sharp)
                t_sharp_train.update(t_sharp)
                s_sharp_train.update(s_sharp)

                loss.backward()
                self.optimizer_student.step()

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                epoch_loss.update(loss.item())
                epoch_ce_loss.update(ce_loss.item())
                epoch_kd_loss.update(kd_loss.item())

            epoch_acc = correct / length_of_dataset

            _, epoch_val_acc, s_sharp_val = self._evaluate_model(self.student_model)
            _, _, t_sharp_val = self._evaluate_model(self.teacher_model)
            self.cfg.VACC['S_LAST'] = epoch_val_acc

            if epoch_val_acc > self.cfg.VACC['S_BEST']:
                self.cfg.VACC['S_BEST'] = epoch_val_acc
                self.best_student_model_weights = deepcopy(self.student_model.state_dict())

            if self.cfg.LOG:
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
                log_cfg(self.cfg)

            print(
                f"[{ep+1}: {(time.time() - t0)/60.0:.1f}m] LR: {self.scheduler_student.get_last_lr()[0]:.1e},",
                f"Loss: {(epoch_loss.avg):.4f}, CE: {(epoch_ce_loss.avg):.4f}, KD: {(epoch_kd_loss.avg):.4f},", 
                f"Acc: {epoch_acc:.4f}, ValAcc: {epoch_val_acc:.4f}")
            print("-" * 100)
            self.scheduler_student.step()

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), self.cfg.SAVE_PATH)

    def train_student(self, save_model=True):
        """
        Function that will be training the student
        """
        if not os.path.exists(self.cfg.TEACHER_WEIGHTS):
            self.train_teacher() 
        else:
            self.teacher_model.load_state_dict(torch.load(self.cfg.TEACHER_WEIGHTS))
        self.cfg.VACC['T_BEST'], _ = self.evaluate(teacher=True)
        print(f"Teacher Accuracy: {self.cfg.VACC['T_BEST']:.4f}%")
        self._train_student(save_model=save_model)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations
        """
        raise NotImplementedError

    def _evaluate_model(self, model):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.
        """
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []
        sharp = AverageMeter()

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                sharp.update(sharpness(output))
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / length_of_dataset
        return outputs, accuracy, sharp.avg

    def evaluate(self, teacher=False, verbose=False):
        """
        Evaluate method for printing accuracies of the trained network
        """
        if teacher:
            model = deepcopy(self.teacher_model).eval().to(self.device)
        else:
            model = deepcopy(self.student_model).eval().to(self.device)
        _, accuracy, sharp = self._evaluate_model(model)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
        return accuracy, sharp
    
    def set_models(self, mode='eval'):
        if mode == 'eval':
            self.teacher_model.eval(), self.student_model.eval()
        elif mode == 'train_teacher':
            self.teacher_model.train(), self.student_model.eval()
        elif mode == 'train_student':
            self.teacher_model.eval(), self.student_model.train()
