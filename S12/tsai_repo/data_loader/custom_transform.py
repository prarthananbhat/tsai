from skimage import transform

class Rescale(object):
    """Resclae the image to the desired shape
    Args:
        output size (tuple or int): Desired output size, If tuple output is matched to output size
        if int then smaller of image edges is matched to outputsize keeping aspect ratio the same
        """
    def __init__(self,output_size):
        # assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target_image = sample['image'],sample['target_image']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w,self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h , new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h, new_w))
        target_img = transform.resize(target_image,(new_h, new_w))

        return {'image':img, 'target_image': target_img}

class ToTensor(object):
    """ converts ndarrays in samples to tensors"""
    def __call__(self, sample):
        image , target_image = sample['image'], sample['target_image']
        image = image.transpose((2,0,1))
        target_image = target_image.transpose((2, 0, 1))
        return image, target_image

