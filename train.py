import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, DML, Shake
# from torchvision.models import resnet18, resnet50
from KD_Lib.models.resnet import ResNet18, ResNet50, ResNet152
from KD_Lib.models.shake import ShakeHead
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Hyperparameters
MODE: str = 'shake' # 'kd' or 'dml' or 'shake'
DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
CLASSES: int = 100
DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
BATCH_SIZE: int = 128
TEACHER = ResNet152
STUDENT = ResNet18
LR: float = 0.1
T: float = 4.0
W: float = 0.5
EPOCHS: int = 200
SCHEDULER: str = 'cos' # 'cos' or 'step'
TEACHER_WEIGHTS: str = f'./models/teacher_{DATASET}.pt'
PARALLEL: bool = False
EXP: str = f"{MODE}_{DATASET}_T"

def main():
    # Dataset
    if DATASET == 'cifar10':
        dataset = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    elif DATASET == 'cifar100':
        dataset = datasets.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    trainset = dataset(root=DATA_PATH, train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testset = dataset(root=DATA_PATH, train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # Models
    teacher_model = TEACHER(num_classes=CLASSES).to('cuda')
    #teacher_model.fc = torch.nn.Linear(2048, CLASSES)
    student_model = STUDENT(num_classes=CLASSES).to('cuda')
    #student_model.fc = torch.nn.Linear(512, CLASSES)
    if PARALLEL:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)
    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    student_optimizer = optim.SGD(student_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    if SCHEDULER == 'cos':
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=200, eta_min=1e-5)
        student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=200, eta_min=1e-5)
    elif SCHEDULER == 'step':
        teacher_scheduler = torch.optim.lr_scheduler.MultiStepLR(teacher_optimizer, milestones=[60, 120, 180], gamma=0.1)
        student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, milestones=[60, 120, 180], gamma=0.1)

    # Training
    if MODE == 'kd': # Vanilla KD
        distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                            teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler,
                            loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                            loss_kd=torch.nn.KLDivLoss(reduction='batchmean'), 
                            temp=T, distil_weight=W, 
                            device="cuda", log=True, logdir=f"./exp/{EXP}/")  
        if not os.path.exists(TEACHER_WEIGHTS):
            distiller.train_teacher(epochs=EPOCHS, plot_losses=True, save_model=True) # Train the teacher network
        else:
            distiller.teacher_model.load_state_dict(torch.load(TEACHER_WEIGHTS))
            distiller.evaluate(teacher=True, verbose=True)
        distiller.train_student(epochs=EPOCHS, plot_losses=True, save_model=True, save_model_path=f"./models/{EXP}.pt") # Train the student network
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

    elif MODE == 'dml': # DML
        students = [teacher_model, student_model]
        optimizers = [teacher_optimizer, student_optimizer]
        schedulers = [teacher_scheduler, student_scheduler]
        distiller = DML(students, train_loader, test_loader, optimizers, schedulers,
                        loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                        loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                        temp=T, distil_weight=W,
                        device="cuda", log=True, logdir=f"./exp/{EXP}/")
        distiller.train_students(epochs=EPOCHS, save_model_path=f"./models/{EXP}.pt")
        distiller.evaluate(verbose=True)

    elif MODE == 'shake': # SHAKE
        data = torch.randn(2, 3, 32, 32).cuda()
        teacher_model.eval()
        _, feat_t, _, _ = teacher_model(data, return_feats=True)
        shake = ShakeHead(feat_t[1:-1]).to('cuda')
        students = [teacher_model, shake, student_model]
        params = list(shake.parameters()) + list(student_model.parameters())
        optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
        if SCHEDULER == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
        elif SCHEDULER == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.1)
        optimizers = [optimizer]
        schedulers = [scheduler]
        distiller = Shake(students, train_loader, test_loader, optimizers, schedulers,
                        loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                        loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                        temp=T, distil_weight=W,
                        device="cuda", log=True, logdir=f"./exp/{EXP}/")
        distiller.train_students(epochs=EPOCHS, save_model_path=f"./models/{EXP}.pt")
        distiller.evaluate(verbose=True)

    else: # Baseline
        model = student_model
        optimizer = student_optimizer
        scheduler = student_scheduler
        distiller = VanillaKD(model, model, train_loader, test_loader, 
                            optimizer, optimizer, scheduler, scheduler,
                            loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), 
                            loss_kd=torch.nn.KLDivLoss(reduction='batchmean'), 
                            temp=T, distil_weight=W, 
                            device="cuda", log=True, logdir=f"./exp/{EXP}/")  
        distiller.train_teacher(epochs=EPOCHS, plot_losses=True, save_model=True, save_model_path=f"./models/{EXP}.pt") # Train the teacher network
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

if __name__ == "__main__":
    main()