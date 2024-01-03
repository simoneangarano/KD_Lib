import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from KD_Lib.KD.common.utils import AverageMeter, sharpness, sharpness_gap
from KD_Lib.KD.vision.DML.dml import DML

class Shake(DML):
    def __init__(
        self,
        students,
        train_loader,
        val_loader,
        optimizers,
        schedulers,
        loss_ce,
        loss_kd,
        temp=1, 
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super().__init__(students, train_loader, val_loader, optimizers, schedulers, loss_ce, loss_kd, temp, distil_weight, device, log, logdir)
        self.C = 100 # Number of classes
        self.ln = torch.nn.LayerNorm((self.C,), elementwise_affine=False, eps=1e-7).to(self.device)

    def train_students(
        self,
        epochs=20,
        save_model=True,
        save_model_path="./models/student_dml.pt",
    ):

        self.students[-1].to(self.device)
        self.students[-1].train()
        self.students[0].to(self.device)
        self.students[0].eval()
        self.students[1].to(self.device)
        self.students[1].train() # ShakeHead

        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        epoch_loss, epoch_ce_loss, epoch_kd_loss = AverageMeter(), AverageMeter(), AverageMeter()
        s_sharp_train, t_sharp_train, g_sharp_train = AverageMeter(), AverageMeter(), AverageMeter()
        self.best_student_model_weights = deepcopy(self.students[0].state_dict())
        self.best_student = self.students[-1]

        print("Training Teacher and Student...")
        for ep in range(epochs):
            t0 = time.time()
            epoch_loss.reset(), epoch_ce_loss.reset(), epoch_kd_loss.reset() 
            s_sharp_train.reset(), t_sharp_train.reset(), g_sharp_train.reset()
            t_correct, s_correct = 0, 0

            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.optimizers:
                    optim.zero_grad()

                logit_s = self.students[-1](data)
                with torch.no_grad():
                    logit_t, feat_t, weight, bias = self.students[0](data, return_feats=True)
                    feat_t = [f.detach() for f in feat_t]

                pred_feat_s = self.students[1](feat_t[2:-1], weight.detach(), bias.detach()) # Feature selection
                logit_s = self.ln(logit_s) *  3.1415
                logit_t = self.ln(logit_t) *  3.1415
                pred_feat_s = self.ln(pred_feat_s) *  3.1415
                # cls + kl div
                loss_cls = self.loss_ce(logit_s, label)
                # loss_div = self.loss_kd(logit_s, logit_t)
                A = self.T * self.T * self.loss_kd(F.log_softmax(pred_feat_s/self.T, dim=1), 
                                                   F.log_softmax(logit_s.detach()/self.T, dim=1))
                B = self.T * self.T * self.loss_kd(F.log_softmax(logit_s/self.T, dim=1), 
                                                   F.log_softmax(pred_feat_s.detach()/self.T, dim=1))
                loss_kd = A + B
                C = self.loss_ce(pred_feat_s, label)
                D = F.mse_loss(pred_feat_s, logit_t.detach())
                loss_kd += C + D
                # print(f"A {A:.4f}, B {B:.4f}, C {C:.4f}, D {D:.4f}")
                loss = (1-self.W) * loss_cls + self.W * loss_kd
                loss.backward()
                self.optimizers[-1].step()

                g_sharp, s_sharp, t_sharp = sharpness_gap(logit_t, logit_s)
                s_sharp_train.update(s_sharp), t_sharp_train.update(t_sharp), g_sharp_train.update(g_sharp)
                epoch_loss.update(loss.item()), epoch_ce_loss.update(loss_cls.item()), epoch_kd_loss.update(loss_kd.item())

                predictions = []
                correct_preds = []
                predictions.append(pred_feat_s.argmax(dim=1, keepdim=True))
                correct_preds.append(predictions[0].eq(label.view_as(predictions[0])).sum().item())
                predictions.append(logit_s.argmax(dim=1, keepdim=True))
                correct_preds.append(predictions[1].eq(label.view_as(predictions[1])).sum().item())
                t_correct += correct_preds[0]
                s_correct += correct_preds[-1]

            t_epoch_acc = t_correct / length_of_dataset
            s_epoch_acc = s_correct / length_of_dataset
            
            val_accs, t_sharp_val, s_sharp_val, g_sharp_val = self.evaluate(verbose=False)
            if val_accs[-1] > best_acc:
                best_acc = val_accs[-1]
                self.best_student_model_weights = deepcopy(self.students[1].state_dict())
                self.best_student = self.students[1]

            if self.log:
                self.writer.add_scalar("Loss Student", epoch_loss.avg, ep)
                self.writer.add_scalar("CE Student", epoch_ce_loss.avg, ep)
                self.writer.add_scalar("KD Student", epoch_kd_loss.avg, ep)
                self.writer.add_scalar("Accuracy Teacher Train", t_epoch_acc, ep)
                self.writer.add_scalar("Accuracy Student Train", s_epoch_acc, ep)
                self.writer.add_scalar("Accuracy Teacher Val", val_accs[0], ep)
                self.writer.add_scalar("Accuracy Student Val", val_accs[-1], ep)
                self.writer.add_scalar("Sharpness Student Train", s_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Student Val", s_sharp_val, ep)
                self.writer.add_scalar("Sharpness Teacher Train", t_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Teacher Val", t_sharp_val, ep)
                self.writer.add_scalar("Sharpness Gap Train", g_sharp_train.avg, ep)
                self.writer.add_scalar("Sharpness Gap Val", g_sharp_val, ep)

            print(f"[{ep+1}: {float(time.time() - t0)/60.0:.1f}m] LR: {self.schedulers[0].get_last_lr()[0]:.1e},",
                  f"Loss: {(epoch_loss.avg):.4f}, CE: {epoch_ce_loss.avg:.4f}, KD: {epoch_kd_loss.avg:.4f}",
                  f"\n[T] Acc: {t_epoch_acc:.4f}, ValAcc: {val_accs[0]:.4f}, [S] Acc: {s_epoch_acc:.4f}, ValAcc: {val_accs[-1]:.4f}")
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
            model = [m.eval() for m in model]
            length_of_dataset = len(self.val_loader.dataset)
            correct = 0
            outputs = []
            sharp = AverageMeter()

            with torch.no_grad():
                for data, target in self.val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    if len(model) == 1:
                        output = model[0](data)
                    else:
                        _, feat_t, weight, bias = model[0](data, return_feats=True)
                        feat_t = [f.detach() for f in feat_t]
                        output = model[1](feat_t[2:-1], weight.detach(), bias.detach()) # Feature selection
                    output = self.ln(output) *  3.1415

                    if isinstance(output, tuple):
                        output = output[0]
                    outputs.append(output)
                    sharp.update(sharpness(output))

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            if verbose:
                print(f"M{name} Accuracy: {correct/length_of_dataset}")

            epoch_val_acc = correct / length_of_dataset
            return outputs, epoch_val_acc, sharp.avg


    def evaluate(self, verbose=False):
        """
        Evaluate method for printing accuracies of the trained student networks

        """        
        models = [deepcopy(m).to(self.device) for m in self.students]
        _, val_acc_t, sharp_t = self._evaluate_model(models[:-1], name="Teacher")
        _, val_acc_s, sharp_s = self._evaluate_model(models[-1:], name="Student")
        val_accs = [val_acc_t, val_acc_s]
        val_sharps = [sharp_t, sharp_s]

        if verbose:
            print(f"Teacher Accuracy: {val_accs[0]:.4f}, Student Accuracy: {val_accs[1]:.4f}")
        return val_accs, val_sharps[0], val_sharps[-1], val_sharps[0] - val_sharps[-1]