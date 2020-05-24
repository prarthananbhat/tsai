#PRINT Loss
import matplotlib.pyplot as plt

class showMeasurePlots():
    def plot_loss_curves(train_losses,test_losses,epochs):
        fig = plt.figure(figsize=(10,4))
        lables = ["Test Loss","Train Loss"]
        plt.plot(test_losses,"red",marker='o')
        plt.plot(train_losses,"green",marker='o')
        plt.title("Loss for {} epochs".format(epochs))
        plt.xlabel("EPOCHS")
        plt.ylabel("LOSS")
        plt.legend(lables)
        plt.show()

    def plot_accuracy_curves(train_accuracy,test_accuracy,epochs):
        fig1 = plt.figure(figsize=(10,4))
        lables = ["Test Accuracy","Train Accuracy"]
        plt.plot(test_accuracy,"red",marker='o')
        plt.plot(train_accuracy,"green",marker='o')
        plt.title("Accuracy for {} epochs".format(epochs))
        plt.xlabel("EPOCHS")
        plt.ylabel("ACCURACY")
        plt.legend(lables)
        plt.show()