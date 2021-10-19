import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def combine_from_dataset(dataset, write_folder):
    img_l_paths = [os.path.join(dataset, scandir, "img_l.png") for scandir in os.listdir(dataset) ]
    proj_paths = [os.path.join(dataset, scandir, "proj.png") for scandir in os.listdir(dataset) ]

    os.makedirs(write_folder, exist_ok=True)

    for i in range(len(img_l_paths)):
        img_l_path = img_l_paths[i]
        proj_path = proj_paths[i]
        img_l = cv2.imread(img_l_path, 0)
        proj = cv2.imread(proj_path, 0)

        stacked = cv2.cvtColor(np.dstack((img_l, proj, np.zeros_like(proj)))*3, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(write_folder, "img"+format(i, "02d")+".png"), stacked)



     



def main():
    print("main")
    datasets = ["test-blurry", "test-specular", "test-pbr"]
    write_ds_top = "double-scan-images"

    for ds in datasets:
        dataset = os.path.join("test-sets", ds)
        write_ds = os.path.join(write_ds_top, ds)
        combine_from_dataset(dataset, write_ds)

        







if __name__ == '__main__':
    main()
