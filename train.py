import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD, DML
# from torchvision.models import resnet18, resnet50
from KD_Lib.models.resnet import ResNet18, ResNet50
torch.autograd.set_detect_anomaly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Hyperparameters
MODE: str = 'dml' # 'kd' or 'dml'
DATASET: str = 'cifar10' # 'cifar10' or 'cifar100'
CLASSES: int = 10
DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
BATCH_SIZE: int = 128
LR: float = 0.1
T: float = 4.0
W: float = 0.5
EPOCHS: int = 200
EXP: str = f"exp/{MODE}/"

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
teacher_model = ResNet50(num_classes=CLASSES)
#teacher_model.fc = torch.nn.Linear(2048, CLASSES)
teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=200, eta_min=1e-5)
student_model = ResNet18(num_classes=CLASSES)
#student_model.fc = torch.nn.Linear(512, CLASSES)
student_optimizer = optim.SGD(student_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=200, eta_min=1e-5)

# Training
if MODE == 'kd':
    # Vanilla KD
    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                          teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler,
                          loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), loss_kd=torch.nn.KLDivLoss(reduction='batchmean'), 
                          temp=T, distil_weight=W, 
                          device="cuda", log=True, logdir=EXP)  
    if not os.path.exists('./models/teacher_kd.pt'):
        distiller.train_teacher(epochs=EPOCHS, plot_losses=True, save_model=True) # Train the teacher network
    else:
        distiller.teacher_model.load_state_dict(torch.load('./models/teacher_kd.pt'))
        distiller.evaluate(teacher=True, verbose=True)
    distiller.train_student(epochs=EPOCHS, plot_losses=True, save_model=True) # Train the student network
    distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network
    distiller.get_parameters() # A utility function to get the number of parameters in the  teacher and the student network

elif MODE == 'dml':
    # DML
    students = [teacher_model, student_model]
    optimizers = [teacher_optimizer, student_optimizer]
    schedulers = [teacher_scheduler, student_scheduler]
    distiller = DML(students, train_loader, test_loader, optimizers, schedulers,
                    loss_ce=torch.nn.CrossEntropyLoss(reduction='mean'), loss_kd=torch.nn.KLDivLoss(reduction='batchmean', log_target=True), 
                    device="cuda", log=True, logdir=EXP)  
    distiller.train_students(epochs=EPOCHS)
    distiller.evaluate(verbose=True)
    distiller.get_parameters()