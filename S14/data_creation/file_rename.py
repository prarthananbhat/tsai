# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    for count, filename in enumerate(os.listdir("C:/Users/pbhat/Google Drive/S14/foreground/")):
        dst = "image_" + str(count) + ".png"
        src = 'C:/Users/pbhat/Google Drive/S14/foreground/' + filename
        dst = 'C:/Users/pbhat/Google Drive/S14/foreground/' + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)

    # Driver Code


if __name__ == '__main__':
    # Calling main() function
    main()