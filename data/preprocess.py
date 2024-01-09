import os
import cv2
from tqdm import tqdm

if __name__ == "__main__":

    root_src = "./mini_train_dota/"
    # root_dst = "./mini_train_dota_preprocessed/"

    for mode in ["train", "test"]:
        
        listdir = os.listdir(f"{root_src}/{mode}/images")
        listdir.sort()

        for file in tqdm(listdir):

            if not file.endswith(".png"):
                continue
            
            # image = cv2.imread("./mini_train_dota/test/images/city_7_0_000292.png", cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(os.path.join(f"{root_src}/{mode}/images", file), cv2.IMREAD_GRAYSCALE)
            
            threshold = 25
            mask = image > threshold
            image = image * mask
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            os.makedirs(f"{root_src}/{mode}/images_preprocessed", exist_ok=True)
            cv2.imwrite(os.path.join(f"{root_src}/{mode}/images_preprocessed", file), image)
