import numpy as np
from skimage import io


def get_tensor_as_image(tensor):
    tensor = tensor[0]
    np_img = tensor.cpu().data.numpy()
    np_img = np.reshape(np_img, (3,150,-1))
    np_img = np.moveaxis(np_img,0,2)
    return np_img

def get_tensor_as_image_grayscale(tensor):
    tensor = tensor[0]
    np_img = tensor.cpu().data.numpy()
    np_img = np.reshape(np_img, (150,-1))
    #np_img = np.moveaxis(np_img,0,2)
    return np.dstack((np_img, np_img, np_img))


def gt_to_cont():
    img = io.imread("/home/ola/projects/masteroppgave/msth-datasets/laser-scan/corner-scan-v1/dataset/ground-truth/img000.png")
    height, width = img.shape
    gt_cont = np.argmax(img, axis=1)
    print(gt_cont.shape)
    print(type(np.argmax(img, axis=1)))
    



    



if __name__ == '__main__':
    gt_to_cont()

