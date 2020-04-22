import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import time
import torchvision
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 32
TOTAL_SIZE =  len(os.listdir('chest_xray/train/NORMAL')) + len(os.listdir('chest_xray/train/PNEUMONIA'))
STEPS_PER_EPOCH =  TOTAL_SIZE//BATCH_SIZE

data_path = 'chest_xray/train/'
    
train_transform = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize((224,224)),
                                                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                torchvision.transforms.RandomRotation(30),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = torchvision.datasets.ImageFolder(
                                         root=data_path,
                                         transform=train_transform
)

train_loader = torch.utils.data.DataLoader(
                                         train_dataset,
                                         batch_size=32,
                                         num_workers=1,
                                         shuffle=True
)



data_path = 'chest_xray/test/'
    
test_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

test_dataset = torchvision.datasets.ImageFolder(
                                         root=data_path,
                                         transform=test_transform
)

test_loader = torch.utils.data.DataLoader(
                                         test_dataset,
                                         batch_size=32,
                                         num_workers=1,
                                         shuffle=True
)

def get_test():
    test_loss = []
    correct = 0
    incorrect = 0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == 19:
            break
        with torch.no_grad():
            data = data.to(device)
            target = target.long().to(device)
            output = model_ft(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output,target)
            for i in range(32):
                a = []
                for j in output[i]:
                    a.append(float(j.detach()))

                pred = a.index(max(a))
        
                if pred == int(target[i]):
                    correct = correct+1

                else:
                    incorrect = incorrect +1

        test_loss.append(float(loss.detach()))
    print(correct,incorrect)
    return "TEST_ACC :- "+str(correct/(correct+incorrect))+" TEST_LOSS:- " +    str(float(sum(test_loss[-9:])/9))

model_ft = torchvision.models.vgg16_bn(pretrained=True)


num_features = model_ft.classifier[6].in_features
features = list(model_ft.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
model_ft.classifier = nn.Sequential(*features) # Replace the model classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.0007 )

loss_history = []

# loss_history = []
for i in range(100):
    sta = time.time()
    print("-----------------------EPOCH "+str(i)+" -----------------------------------")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == STEPS_PER_EPOCH:
            break
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model_ft(data)
        loss = criterion(output,target.reshape((BATCH_SIZE,)).long())
        loss_history.append(loss.detach())
        
        correct = 0
        incorrect =0
        for p in range(32):
            a = []
            for j in output[p]:
                a.append(float(j.detach()))

            pred = a.index(max(a))
        
            if pred == int(target[p]):
                correct = correct+1

            else:
                
                incorrect = incorrect +1

        print("EPOCH "+str(i)+' MINIBATCH :- '+str(batch_idx)+"/"+str(STEPS_PER_EPOCH)+" LOSS : - "+str(loss_history[-1])+" ACC:- "+str(correct/(incorrect+correct)),end = "\r")



        loss.backward()
        optimizer.step()
    stb = time.time()
    print("EPOCH "+str(i)+" LOSS "+str(sum(loss_history[-STEPS_PER_EPOCH:])/STEPS_PER_EPOCH)+" ETA:- "+str(stb-sta) + " \n MAX LOSS:- "+str(max(loss_history[-STEPS_PER_EPOCH:]))+" MIN LOSS:- "+str(min(loss_history[-STEPS_PER_EPOCH:])))
    torch.save(model_ft.state_dict(), 'vgg16v1-acc.pt')

get_test()
