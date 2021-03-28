
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


