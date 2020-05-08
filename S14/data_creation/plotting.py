import matplotlib.pyplot as plt
def showimage(img):
    print("Size:",img.shape)
    plt.imshow(img,cmap="gray")
    plt.show()

def showpreprocessed(bg_img,fg_img,fg_img_mask,merged_img,mask_img):
    fig = plt.figure(figsize=(20,10))
    img = bg_img
    ax1 = fig.add_subplot(1,5,1)
    ax1.imshow(img)
    img = fg_img
    ax2 = fig.add_subplot(1,5,2)
    ax2.imshow(img)
    img = fg_img_mask
    ax3 = fig.add_subplot(1,5,3)
    ax3.imshow(img,cmap="gray")
    img = merged_img
    ax4 = fig.add_subplot(1,5,4)
    ax4.imshow(img)
    img = mask_img
    ax5 = fig.add_subplot(1,5,5)
    ax5.imshow(img,cmap="gray")
    plt.show()