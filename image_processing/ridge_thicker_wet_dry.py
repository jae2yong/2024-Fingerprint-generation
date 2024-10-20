import cv2
import numpy as np
import os

def erode_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # erosion 연산을 위한 kernel 정의
    kernel = np.ones((3,3), np.uint8)

    # erosion 연산 수행
    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(image, kernel, iterations=1)
    # 결과 이미지 저장
    # cv2.imwrite(image_path.replace('_O_N_', '_O_W_').replace('.BMP', '.bmp'), eroded)
    # cv2.imwrite(image_path.replace('_O_N_', '_O_D_').replace('.BMP', '.bmp'), dilated)
    cv2.imwrite(image_path.replace('_C_N_', '_C_W_'),eroded)
    cv2.imwrite(image_path.replace('_C_N_', '_C_D_'),dilated)
# 지정된 폴더 내에서 'O_N_1'이 포함된 모든 이미지에 대해 erosion 연산 적용
root_folder = r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\13_Styletransfer_AdaIN_pytorch\Pytorch_Adain_from_scratch\0205_Dfinger_capacitive_bright'  # 여기에 원하는 폴더 경로를 입력하세요.

for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if any(filename.endswith(f'_C_N_{i}.BMP') for i in range(1, 6)):
            image_path = os.path.join(foldername, filename)
            erode_image(image_path)