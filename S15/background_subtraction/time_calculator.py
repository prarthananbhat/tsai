import timeit

def data_read_time():
    SETUP_CODE = '''import os
from custom_dataset import imageMaskBackgroundDataset
dataroot = "D:/Projects/theschoolofai/datasets/background_subtraction/mini_multiple_bg_multi_foreground/"
trainroot = os.path.join(dataroot,"train_data")
testroot = os.path.join(dataroot,"test_data")'''

    TEST_CODE = '''
imageMaskBackgroundDataset(trainroot+"/train.csv",dataroot)
    '''
    times = timeit.repeat(setup= SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat = 3000,
                          number = 100)

    # priniting minimum exec. time
    print('image data read time: {}'.format(min(times)))
    print('image data read time: {}'.format(times))

if __name__ == "__main__":
    data_read_time()