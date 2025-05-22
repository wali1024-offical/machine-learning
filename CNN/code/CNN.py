import numpy as np
from numpy.random import logistic
import pandas as pd
import random
import torch
import torch.nn as nn   #神经网络的基本模块
from torch.utils.data import Dataset,DataLoader,Subset, ConcatDataset #1.用来继承从而构建自己的数据集 2.把dataset封装起来自动打包成类 3.用来划分验证集和训练集4.拼接数据集
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder,VisionDataset
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

#设置随机种子 保证结果可以复现
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


#对数据集进行处理
test_tfm = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),])
train_tfm = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),])

#构建数据集


class FoodDataset(Dataset):  
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset, self).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        filename = Path(fname).name
        if "_" in filename:
            label = int(filename.split("_")[0])
            assert 0 <= label < 11, f"非法标签: {label}, 文件: {fname}"
        else:
            label = -1

        return im, label


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), 
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )    

    def forward(self,x):
            out = self.cnn(x)
            out = out.view(out.size()[0],-1)
            return self.fc(out)

batch_size = 64
_dataset_dir = r"F:\machine_learning\2022\3\dataset"
train_set = FoodDataset(os.path.join(_dataset_dir, "training"), tfm=train_tfm)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True )
valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)



device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 4
patience = 300

model = Classifier().to(device)  #
criterion = nn.CrossEntropyLoss()  #
optimizer = torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-5)  #

stale = 0    #
best_acc = 0 #

_exp_name = "sample"  # 或你想要的任意名称
writer = SummaryWriter(log_dir=f"runs/{_exp_name}")
for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch
        logists = model(imgs.to(device))
        loss = criterion(logists, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        acc = (logists.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()  # 设置为评估模式（关闭 Dropout、BatchNorm 的更新）

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logists = model(imgs.to(device))

        loss = criterion(logists, labels.to(device))
        acc = (logists.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    writer.add_scalar("Loss/valid", valid_loss, epoch)
    writer.add_scalar("Accuracy/valid", valid_acc, epoch)

    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
      
    
    
    # 记录日志
    if valid_acc > best_acc:

        with open(f"F:/machine_learning/2022/3/logs/{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"F:/machine_learning/2022/3/logs/{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # 保存模型
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"F:/machine_learning/2022/3/logs/{_exp_name}_best.ckpt")  # 保存权重
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement {patience} consecutive epochs, early stopping")
            break

writer.close()
 
test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


model_best = Classifier().to(device)  # 初始化模型结构并放到 device 上
model_best.load_state_dict(torch.load(f"F:/machine_learning/2022/3/logs/{_exp_name}_best.ckpt"))  # 加载最佳权重
model_best.eval()  # 设置为推理模式

prediction = []

with torch.no_grad():  # 关闭梯度追踪
    for data, _ in test_loader:
        test_pred = model_best(data.to(device))  # 前向传播，预测结果
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)  # 取预测最大值对应的类别
        prediction += test_label.squeeze().tolist()  # 累加进 prediction 列表中


# 辅助函数：补足4位编号，例如 1 → "0001"
def pad4(i):
    return "0"*(4 - len(str(i))) + str(i)

# 创建空 DataFrame
df = pd.DataFrame()

# 写入测试图像的 ID（从 "0001" 到 len(test_set)）
df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]

# 写入预测结果
df["Category"] = prediction

# 导出为 CSV 文件（不写 index）
df.to_csv("submission.csv", index=False)





            


            


        




        






     









