import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from KD_Lib.utils import AverageMeter, sharpness, sharpness_gap, log_cfg
from KD_Lib.KD.vision.DML.shake import Shake

class TriKD(Shake):
    def __init__(self, models, loaders, optimizers, schedulers, losses, cfg):
        super().__init__(models, loaders, optimizers, schedulers, losses, cfg)
    
    def train_student(self, save_model=True):
        length_of_dataset = len(self.train_loader.dataset)
        epoch_loss, epoch_ce_loss, epoch_kd_loss = AverageMeter(), AverageMeter(), AverageMeter()
        s_sharp_train, t_sharp_train, g_sharp_train = AverageMeter(), AverageMeter(), AverageMeter()
        self.best_student_model_weights = deepcopy(self.models[-1].state_dict())
        self.best_student = self.models[-1]

        self.logger.save_log("Training Teacher and Student...")
        self.cfg.TIME = time.time()
        for ep in range(self.cfg.EPOCHS):
            if ep == 150:
                self.cfg.W = 100
            t0 = time.time()
            epoch_loss.reset(), epoch_ce_loss.reset(), epoch_kd_loss.reset() 
            s_sharp_train.reset(), t_sharp_train.reset(), g_sharp_train.reset()
            t_correct, s_correct = 0, 0
            self.set_models(mode='train_student')
            
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.optimizers:
                    optim.zero_grad()

                logit_s = self.models[-1](data, norm_feats=self.cfg.FEAT_NORM)

                with torch.no_grad():
                    logit_a = self.models[0](data, norm_feats=self.cfg.FEAT_NORM)
                    logit_t, feat_t, _, _ = self.models[1](data, return_feats=True, norm_feats=self.cfg.FEAT_NORM)
                    feat_t = [f.detach() for f in feat_t]

                if isinstance(logit_t, tuple):
                    logit_t_orig, logit_t = logit_t
                    logit_s_orig, logit_s = logit_s
                    _, logit_a = logit_a
                else:
                    logit_t_orig = logit_t
                    logit_s_orig = logit_s

                pred_feat_s = self.models[-2](feat_t[-1]) # Feature selection
                # classification loss student
                loss_cls = self.loss_ce(logit_s, label)
                # distillation loss student <-> smooth head
                A = self.cfg.T * self.cfg.T * self.loss_kd(F.log_softmax(pred_feat_s/self.cfg.T, dim=1), 
                                                           F.log_softmax(logit_s.detach()/self.cfg.T, dim=1))
                B = self.cfg.T * self.cfg.T * self.loss_kd(F.log_softmax(logit_s/self.cfg.T, dim=1), 
                                                           F.log_softmax(pred_feat_s.detach()/self.cfg.T, dim=1))
                # classification loss smooth head
                # C = self.loss_ce(pred_feat_s, label)
                # distillation loss smooth head <-> teacher
                # D = F.mse_loss(pred_feat_s, logit_t.detach())
                # distillation loss student <-> frozen teacher
                # E = self.cfg.T * self.cfg.T * self.loss_kd(F.log_softmax(logit_s/self.cfg.T, dim=1), 
                #                                            F.log_softmax(logit_t.detach()/self.cfg.T, dim=1))
                # distillation loss smooth teacher <-> anchor
                G = self.cfg.T * self.cfg.T * self.loss_kd(F.log_softmax(pred_feat_s/self.cfg.T, dim=1), 
                                                           F.log_softmax(logit_a/self.cfg.T, dim=1))
                # distillation loss student <-> anchor
                H = self.cfg.T * self.cfg.T * self.loss_kd(F.log_softmax(logit_s/self.cfg.T, dim=1), 
                                                           F.log_softmax(logit_a/self.cfg.T, dim=1))
                                                           
                loss_kd = A + B + G + H # + D + E + C
                loss = loss_cls + self.cfg.W * loss_kd
                loss.backward()
                self.optimizers[-1].step()

                g_sharp, t_sharp, s_sharp = sharpness_gap(pred_feat_s, logit_s_orig)
                s_sharp_train.update(s_sharp), t_sharp_train.update(t_sharp), g_sharp_train.update(g_sharp)
                epoch_loss.update(loss.item()), epoch_ce_loss.update(loss_cls.item()), epoch_kd_loss.update(loss_kd.item())

                predictions = []
                correct_preds = []
                predictions.append(pred_feat_s.argmax(dim=1, keepdim=True))
                correct_preds.append(predictions[0].eq(label.view_as(predictions[0])).sum().item())
                predictions.append(logit_s_orig.argmax(dim=1, keepdim=True))
                correct_preds.append(predictions[1].eq(label.view_as(predictions[1])).sum().item())
                t_correct += correct_preds[0]
                s_correct += correct_preds[-1]

            t_epoch_acc = t_correct / length_of_dataset
            s_epoch_acc = s_correct / length_of_dataset
            
            val_accs, t_sharp_val, s_sharp_val, g_sharp_val = self.evaluate(verbose=False)
            self.cfg.VACC['T_LAST'] = val_accs[0]
            self.cfg.VACC['S_LAST'] = val_accs[-1]
            if val_accs[-1] > self.cfg.VACC['S_BEST']:
                self.cfg.VACC['S_BEST'] = val_accs[-1]
                self.cfg.VACC['T_BEST'] = val_accs[0]
                self.best_student_model_weights = deepcopy(self.models[-1].state_dict())
                self.best_student = self.models[-1]

            if self.cfg.LOG:
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
                log_cfg(self.cfg)

            self.logger.save_log(f"[{ep+1}: {float(time.time() - t0)/60.0:.1f}m] LR: {self.schedulers[-1].get_last_lr()[0]:.1e},",
                  f"Loss: {(epoch_loss.avg):.4f}, CE: {epoch_ce_loss.avg:.4f}, KD: {epoch_kd_loss.avg:.4f}",
                  f"\n[T] Acc: {t_epoch_acc:.4f}, ValAcc: {val_accs[0]:.4f}, [S] Acc: {s_epoch_acc:.4f}, ValAcc: {val_accs[-1]:.4f}")
            self.logger.save_log("-" * 60)
            self.schedulers[-1].step()

        self.cfg.TIME = (time.time() - self.cfg.TIME) / 60.0
        self.best_student.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.best_student.state_dict(), self.cfg.SAVE_PATH)

    def evaluate(self, verbose=False, teacher=False, anchor=False):
        """
        Evaluate method for printing accuracies of the trained student networks

        """        
        models = [deepcopy(m).eval().to(self.device) for m in self.models]
        if teacher:
            _, val_acc_t, sharp_t = self._evaluate_model(models[1:2])
            return val_acc_t, sharp_t
        if anchor:
            _, val_acc_a, sharp_a = self._evaluate_model(models[0:1])
            return val_acc_a, sharp_a
        _, val_acc_t, sharp_t = self._evaluate_model(models[1:3])
        _, val_acc_s, sharp_s = self._evaluate_model(models[-1:])
        val_accs = [val_acc_t, val_acc_s]
        val_sharps = [sharp_t, sharp_s]
        if verbose:
            self.logger.save_log(f"Teacher Accuracy: {val_accs[0]:.4f}, Student Accuracy: {val_accs[1]:.4f}")
        return val_accs, val_sharps[0], val_sharps[-1], val_sharps[0] - val_sharps[-1]
    
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
                    if len(model) == 1:
                        output = model[0](data)
                    else:
                        _, feat_t, _, _ = model[0](data, return_feats=True)
                        feat_t = [f.detach() for f in feat_t]
                        output = model[-1](feat_t[-1]) # Feature selection
                    outputs.append(output)
                    sharp.update(sharpness(output))

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            epoch_val_acc = correct / length_of_dataset
            return outputs, epoch_val_acc, sharp.avg

