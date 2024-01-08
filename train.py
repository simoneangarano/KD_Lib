import os, json
from pprint import pformat
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, DML, Shake, Smooth
from torchvision.models import resnet18, resnet50, resnet152
from KD_Lib.models.resnet import ResNet18, ResNet50, ResNet152
from KD_Lib.models.shake import ShakeHead
from datasets import Cub200
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Hyperparameters
class Cfg:
    def __init__(self, dict=None):
        if dict is not None:
            for key in dict:
                setattr(self, key, dict[key])
            return
        
        self.MODE: str = 'kd' # 'kd' or 'dml' or 'shake'
        self.DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
        self.CLASSES: int = 100
        self.DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
        self.BATCH_SIZE: int = 128
        self.TEACHER: str = 'resnet152' 
        self.STUDENT: str = 'resnet18'
        self.LAYER_NORM: bool = False
        self.LR: float = 0.1
        self.LR_MIN: float = 1e-5 #1e-5
        self.T: float = 1.0
        self.W: float = 0.5
        self.EPOCHS: int = 200
        self.SCHEDULER: str = 'step' # 'cos' or 'step' or 'lin'
        self.TEACHER_WEIGHTS: str = f'./models/teacher_{self.DATASET}_{self.MODE}.pt'
        self.PARALLEL: bool = False
        self.EXP: str = f"{self.MODE}_{self.DATASET}_new"

def main():
    cfg = Cfg()
    print(pformat(cfg.__dict__))
    if not os.path.exists(f"./exp/{cfg.EXP}"):
        os.makedirs(f"./exp/{cfg.EXP}")
    with open(f"./exp/{cfg.EXP}/cfg.json", "w") as file:
        json.dump(cfg.__dict__, file)

    # Dataset
    if cfg.DATASET == 'cifar10':
        dataset = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        imsize = 32
    elif cfg.DATASET == 'cifar100':
        dataset = datasets.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)
        imsize = 32
    elif dataset == 'cub200':
        mean=[104/255.0, 117/255.0, 128/255.0],
        std= [1/255.0, 1/255.0, 1/255.0]
        imsize = 227
        dataset = Cub200

    train_transform = transforms.Compose([
        transforms.RandomCrop(imsize, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = dataset(root=cfg.DATA_PATH, train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testset = dataset(root=cfg.DATA_PATH, train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # Models
    cfg.TEACHER = ResNet152 if cfg.TEACHER == 'resnet152' else ResNet50 if cfg.TEACHER == 'resnet50' else ResNet18
    teacher_model = cfg.TEACHER(num_classes=cfg.CLASSES).to('cuda')
    #teacher_model.fc = torch.nn.Linear(2048, CLASSES)
    cfg.STUDENT = ResNet152 if cfg.STUDENT == 'resnet152' else ResNet50 if cfg.STUDENT == 'resnet50' else ResNet18
    student_model = cfg.STUDENT(num_classes=cfg.CLASSES).to('cuda')
    #student_model.fc = torch.nn.Linear(512, CLASSES)
    if cfg.PARALLEL:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)
    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=cfg.LR, momentum=0.9, weight_decay=1e-4)
    student_optimizer = optim.SGD(student_model.parameters(), lr=cfg.LR, momentum=0.9, weight_decay=1e-4)
    if cfg.SCHEDULER == 'cos':
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
        student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
    elif cfg.SCHEDULER == 'step':
        teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optimizer, milestones=[60, 120, 180], gamma=0.1)
        student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, milestones=[60, 120, 180], gamma=0.1)
    elif cfg.SCHEDULER == 'lin':
        teacher_scheduler = torch.optim.lr_scheduler.LinearLR(teacher_optimizer, total_iters=cfg.EPOCHS, start_factor=1, end_factor=cfg.LR_MIN/cfg.LR)
        student_scheduler = torch.optim.lr_scheduler.LinearLR(student_optimizer, total_iters=cfg.EPOCHS, start_factor=1, end_factor=cfg.LR_MIN/cfg.LR)

    # Training
    if cfg.MODE == 'kd': # Vanilla KD
        distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                            teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler,
                            loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                            loss_kd=torch.nn.KLDivLoss(reduction='batchmean'), 
                            temp=cfg.T, distil_weight=cfg.W, 
                            device="cuda", log=True, logdir=f"./exp/{cfg.EXP}/")  
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher(epochs=cfg.EPOCHS, save_model=True, save_model_path=cfg.TEACHER_WEIGHTS) # Train the teacher network
        else:
            distiller.teacher_model.load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
            t_val, _ = distiller.evaluate(teacher=True)
            print(f"Teacher Accuracy: {t_val:.4f}%")
        distiller.train_student(epochs=cfg.EPOCHS, plot_losses=True, save_model=True, save_model_path=f"./models/{cfg.EXP}.pt") # Train the student network
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

    elif cfg.MODE == 'dml': # DML
        students = [teacher_model, student_model]
        optimizers = [teacher_optimizer, student_optimizer]
        schedulers = [teacher_scheduler, student_scheduler]
        distiller = DML(students, train_loader, test_loader, optimizers, schedulers,
                        loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                        loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                        temp=cfg.T, distil_weight=cfg.W,
                        device="cuda", log=True, logdir=f"./exp/{cfg.EXP}/")
        distiller.train_students(epochs=cfg.EPOCHS, save_model_path=f"./models/{cfg.EXP}.pt")
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'shake': # SHAKE
        data = torch.randn(2, 3, 32, 32).cuda()
        teacher_model.eval()
        _, feat_t, _, _ = teacher_model(data, return_feats=True)
        shake = ShakeHead(feat_t[1:-1]).to('cuda')
        students = [teacher_model, shake, student_model]
        params = list(shake.parameters()) + list(student_model.parameters())
        optimizer = optim.SGD(params, lr=cfg.LR, momentum=0.9, weight_decay=5e-4)
        if cfg.SCHEDULER == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
        elif cfg.SCHEDULER == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
        optimizers = [optimizer]
        schedulers = [scheduler]
        distiller = Shake(students, train_loader, test_loader, optimizers, schedulers,
                        loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                        loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                        temp=cfg.T, distil_weight=cfg.W, layer_norm=cfg.LAYER_NORM,
                        device="cuda", log=True, logdir=f"./exp/{cfg.EXP}/")
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            t_optimizer = optim.SGD(distiller.students[0].parameters(), lr=cfg.LR, momentum=0.9, weight_decay=5e-4)
            t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(t_optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
            distiller.train_teacher(optimizer=t_optimizer, scheduler=t_scheduler,
                                    epochs=cfg.EPOCHS, save_model=True, save_model_path=cfg.TEACHER_WEIGHTS)
        else:
            distiller.students[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        t_val, _ = distiller.evaluate(teacher=True)
        print(f"Teacher Accuracy: {t_val:.4f}%")
        distiller.train_students(epochs=cfg.EPOCHS, save_model_path=f"./models/{cfg.EXP}.pt")
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'smooth': # New method
        smooth = torch.nn.Linear(2048, cfg.CLASSES).to('cuda')
        smooth.weight.data = teacher_model.linear.weight.data.clone()
        students = [teacher_model, smooth, student_model]
        params = list(smooth.parameters()) + list(student_model.parameters())
        optimizer = optim.SGD(params, lr=cfg.LR, momentum=0.9, weight_decay=5e-4)
        if cfg.SCHEDULER == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
        elif cfg.SCHEDULER == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
        optimizers = [optimizer]
        schedulers = [scheduler]
        distiller = Smooth(students, train_loader, test_loader, optimizers, schedulers,
                        loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                        loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                        temp=cfg.T, distil_weight=cfg.W,
                        device="cuda", log=True, logdir=f"./exp/{cfg.EXP}/")
        distiller.students[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        t_val, _ = distiller.evaluate(teacher=True)
        print(f"Teacher Accuracy: {t_val:.4f}%")
        distiller.train_students(epochs=cfg.EPOCHS, save_model_path=f"./models/{cfg.EXP}.pt")
        distiller.evaluate(verbose=True)

    else: # Baseline
        model = student_model
        optimizer = student_optimizer
        scheduler = student_scheduler
        distiller = VanillaKD(model, model, train_loader, test_loader, 
                            optimizer, optimizer, scheduler, scheduler,
                            loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                            loss_kd=torch.nn.KLDivLoss(reduction='batchmean'), 
                            temp=cfg.T, distil_weight=cfg.W, 
                            device="cuda", log=True, logdir=f"./exp/{cfg.EXP}/")  
        distiller.train_teacher(epochs=cfg.EPOCHS, plot_losses=True, save_model=True, save_model_path=f"./models/{cfg.EXP}.pt") # Train the teacher network
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

if __name__ == "__main__":
    main()