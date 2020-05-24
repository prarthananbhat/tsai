from tqdm import tqdm
import torch.nn as nn
import torch
import cv2

def train(model, device, train_loader, optimizer, train_losses, train_accuracy, L1lambda, scheduler):
    print(device)
    correct = 0
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Calculate loss
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target)

        # Implementing L1 regularization
        if L1lambda > 0:
          reg_loss = 0.
          for param in model.parameters():
            reg_loss += torch.sum(param.abs())
          loss += L1lambda * reg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss)
        # scheduler.step(loss)

        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()



    train_losses.append(loss)
    train_accuracy.append(100. * correct / len(train_loader.dataset))

def custom_train(model, device, train_loader, classes, optimizer, train_losses, train_accuracy):
    main_path = "D:/Projects/theschoolofai/datasets/custom_dataset/"
    target_img_path = main_path + "target/"
    ext = ".jpg"
    print(device)
    correct = 0
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        print(target.shape)
        print(target)
        target_img = cv2.imread(target_img_path + classes[target] + ext)
        print(target_img.shape)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Calculate loss
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target)
        # Backpropagation
        loss.backward()
        optimizer.step()
        pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_losses.append(loss)
    train_accuracy.append(100. * correct / len(train_loader.dataset))

