import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from KD_Lib.utils import AverageMeter, sharpness, sharpness_gap, log_cfg

class DML:
    """
    Implementation of "Deep Mutual Learning" https://arxiv.org/abs/1706.00384
    """

    def __init__(self, models, loaders, optimizers, schedulers, losses, cfg):
        self.models = models
        self.train_loader, self.val_loader = loaders
        self.optimizers = optimizers
        self.schedulers = schedulers
        if len(losses) == 2:
            self.loss_ce, self.loss_kd = losses
        else:
            self.loss_ce, self.loss_kd, self.loss_sharp = losses

        self.cfg = cfg
        self.device = torch.device(self.cfg.DEVICE)

        if self.cfg.LOG:
            self.writer = SummaryWriter(self.cfg.TB_DIR)
            layout = {"Metrics": {"Loss": ["Multiline", ["Loss Teacher", "Loss Student", "CE Student", "KD Student"]],
                                  "Accuracy": ["Multiline", ["Accuracy Teacher Train", "Accuracy Student Train", 
                                                             "Accuracy Student Val", "Accuracy Teacher Val"]],
                                  "Sharpness": ["Multiline", ["Sharpness Teacher Train", "Sharpness Student Train",
                                                              "Sharpness Student Val", "Sharpness Teacher Val"]],
                                  "Sharpness Gap": ["Multiline", ["Sharpness Gap Train", "Sharpness Gap Val"]],
                                  "KL Divergence": ["Multiline", ["KL Divergence"]]}}
            self.writer.add_custom_scalars(layout)

    def train_student(self, save_model=True, stud_teach_kd=True):
        num_students = len(self.models)
        length_of_dataset = len(self.train_loader.dataset)
        epoch_loss, epoch_ce_loss, epoch_kd_loss = AverageMeter(), AverageMeter(), AverageMeter()
        s_sharp_train, t_sharp_train, g_sharp_train = AverageMeter(), AverageMeter(), AverageMeter()
        epoch_kld = AverageMeter()
        self.best_student_model_weights = deepcopy(self.models[-1].state_dict())
        self.best_student = self.models[-1]

        self.logger.save_log("Training Teacher and Student...")
        self.cfg.TIME = time.time()
        for ep in range(self.cfg.EPOCHS):
            t0 = time.time()
            epoch_loss.reset(), epoch_ce_loss.reset(), epoch_kd_loss.reset() 
            s_sharp_train.reset(), t_sharp_train.reset(), g_sharp_train.reset()
            epoch_kld.reset()
            t_correct, s_correct = 0, 0
            self.set_models(mode='train')

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
                        out_i = self.models[i](data, norm_feats=self.cfg.FEAT_NORM)
                        with torch.no_grad():
                            out_j = self.models[j](data, norm_feats=self.cfg.FEAT_NORM)
                        if stud_teach_kd or i == 1: # Student-Teacher KD skipped if stud_teach_kd is False
                            loss_kd = self.loss_kd(out_i, out_j)
                            student_loss += self.cfg.W*loss_kd
                    student_loss /= num_students - 1
                    loss_ce = self.loss_ce(out_i, label)
                    student_loss += loss_ce
                    student_loss.backward()
                    self.optimizers[i].step()

                g_sharp, t_sharp, s_sharp = sharpness_gap(out_j, out_i) # j is teacher, i is student
                s_sharp_train.update(s_sharp), t_sharp_train.update(t_sharp), g_sharp_train.update(g_sharp)
                epoch_loss.update(student_loss.item()), epoch_ce_loss.update(loss_ce.item()), epoch_kd_loss.update(loss_kd.item())
                kld = F.kl_div(F.log_softmax(out_i.detach(), dim=1), F.softmax(out_j.detach(), dim=1), reduction='batchmean')
                epoch_kld.update(kld)

                predictions = []
                correct_preds = []
                predictions.append(out_j.argmax(dim=1, keepdim=True))
                correct_preds.append(predictions[0].eq(label.view_as(predictions[0])).sum().item())
                predictions.append(out_i.argmax(dim=1, keepdim=True))
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
                self.writer.add_scalar("KL Divergence", epoch_kld.avg, ep)
                log_cfg(self.cfg)

            self.logger.save_log(f"[{ep+1}: {(time.time() - t0)/60.0:.1f}m] LR: {self.schedulers[0].get_last_lr()[0]:.1e},",
                  f"Loss: {(epoch_loss.avg):.4f}, CE: {epoch_ce_loss.avg:.4f}, KD: {epoch_kd_loss.avg:.4f}",
                  f"\n[T] Acc: {t_epoch_acc:.4f}, ValAcc: {val_accs[0]:.4f}, [S] Acc: {s_epoch_acc:.4f}, ValAcc: {val_accs[-1]:.4f}")
            self.logger.save_log("-" * 60)
            for sched in self.schedulers:
                sched.step()
                
        self.cfg.TIME = (time.time() - self.cfg.TIME) / 60.0
        self.best_student.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.best_student.state_dict(), self.cfg.SAVE_PATH)

    def _evaluate_model(self, model):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.
        """
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []
        sharp = AverageMeter()
        model.eval()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                outputs.append(output)
                sharp.update(sharpness(output))

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        epoch_val_acc = correct / length_of_dataset
        return outputs, epoch_val_acc, sharp.avg

    def evaluate(self, verbose=False):
        """
        Evaluate method for printing accuracies of the trained student networks
        """
        val_accs, val_sharps = [], []
        for model in self.models:
            model = deepcopy(model).eval().to(self.device)
            _, val_acc_i, sharp = self._evaluate_model(model)
            val_accs.append(val_acc_i)
            val_sharps.append(sharp)
        if verbose:
            self.logger.save_log(f"Teacher Accuracy: {val_accs[0]:.4f}, Student Accuracy: {val_accs[1]:.4f}")
        return val_accs, val_sharps[0], val_sharps[-1], val_sharps[0] - val_sharps[-1]
    
    def set_models(self, mode='eval'):
        if mode == 'eval':
            [model.eval() for model in self.models]
        elif mode == 'train_teacher':
            self.models[0].train(), self.models[-1].eval()
        elif mode == 'train_student':
            self.models[0].eval(), self.models[-1].train()
        elif mode == 'train':
            [model.train() for model in self.models]
        else:
            raise ValueError(f"Mode {mode} not supported.")