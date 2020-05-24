import torch

class identifyImages:
    def misclassified(model, test_loader, device):
        with torch.no_grad():
            for_the_first_time = 1
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                predicted_val = pred[target.view_as(pred) != pred]
                actual_val = target.view_as(pred)[target.view_as(pred) != pred]
                misclass_img = data[(target.view_as(pred) != pred).view_as(target)]

                if for_the_first_time == 1:
                    misclass_data = misclass_img
                    misclass_targets = actual_val
                    misclass_preds = predicted_val
                    for_the_first_time = 0
                elif for_the_first_time == 0:
                    # one mis classified image already found now just append anymore
                    misclass_data = torch.cat([misclass_data, misclass_img], dim=0)
                    misclass_targets = torch.cat([misclass_targets, actual_val], dim=0)
                    misclass_preds = torch.cat([misclass_preds, predicted_val], dim=0)
                    if len(misclass_data > 25):
                        break
                else:
                    print("No Mis Classifications")
        return misclass_data, misclass_targets, misclass_preds


    def correct_classified(model, test_loader, device):
        with torch.no_grad():
            for_the_first_time = 1
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                predicted_val = pred[target.view_as(pred) == pred]
                actual_val = target.view_as(pred)[target.view_as(pred) == pred]
                correctclass_img = data[(target.view_as(pred) == pred).view_as(target)]
                if for_the_first_time == 1:
                    correctclass_data = correctclass_img
                    correctclass_targets = actual_val
                    correctclass_preds = predicted_val
                    for_the_first_time = 0
                elif for_the_first_time == 0:
                    # one mis classified image already found now just append anymore
                    correctclass_data = torch.cat([correctclass_data, correctclass_img], dim=0)
                    correctclass_targets = torch.cat([correctclass_targets, actual_val], dim=0)
                    correctclass_preds = torch.cat([correctclass_preds, predicted_val], dim=0)
                    if len(correctclass_data > 25):
                        break
                else:
                    print("No Mis Classifications")
        return correctclass_data, correctclass_targets, correctclass_preds