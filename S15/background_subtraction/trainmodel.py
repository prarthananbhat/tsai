from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable
import timeit
class trainModel():
    def __init__(self):
        pass

    def train(self, model, device, train_loader, optimizer, train_losses, train_batch_timer):
        model=model.float()
        pbar = tqdm(train_loader)
        batch_train_start_time = list()
        for batch_idx, (data, target, bg) in enumerate(pbar):
            batch_train_start = timeit.default_timer()
            #Add the Background image
            # print("Data Shape:",data.size())
            # print("Background Shape:",bg.size())
            data_processing_time = timeit.default_timer()
            data = torch.cat((data, bg), 1)
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)
            data_processing_elpase = timeit.default_timer() - data_processing_time

            # print(data.shape, target.shape)
            forward_pass_time = timeit.default_timer()
            optimizer.zero_grad()
            output = model(data.float())
            forward_pass_elapse = timeit.default_timer()- forward_pass_time
            # Calculate loss
            loss_time = timeit.default_timer()
            criteria = nn.MSELoss()
            loss = criteria(output, target.float())
            loas_time_elapse = timeit.default_timer() - loss_time
            back_prop_time = timeit.default_timer()
            # Backpropagation
            loss.backward()
            optimizer.step()
            back_prop_elapse = timeit.default_timer() - back_prop_time
            pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')
            batch_train_end = timeit.default_timer()
            batch_train_elapse = (batch_train_end-batch_train_start)
            batch_train_start_time.append([batch_train_elapse
                                              ,loas_time_elapse
                                              ,back_prop_elapse
                                              ,forward_pass_elapse
                                              ,data_processing_elpase])
        train_losses.append(loss)
        train_batch_timer.append(batch_train_start_time)
        return train_losses, train_batch_timer

