import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import show_image_batch
class testModel:
    def __init__(self):
        pass

    def test(self,model, device, test_loader, test_losses, test_batch_timer):
        test_loss = 0

        with torch.no_grad():
            for data, target, bg in test_loader:
                data = torch.cat((data, bg), 1)
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                output = model(data.float())
                target = target.unsqueeze(1)
                distance = nn.MSELoss(reduction="mean")
                test_loss = distance(output, target.float())
            test_losses.append(test_loss)
            #Showing the last batch
            print("Showing the Last Batch")
            show_image_batch(output = output, target = target)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        return test_losses