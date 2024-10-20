- PyCharm
- Python 3.5
- Tensorflow CPU 1.9
- Other packages:
- numpy
- scipy
- keras
- pillow
- tkinter
- pythonnet

**1. Diffusion(thumbnail)**
참고 : https://huggingface.co/docs/diffusers/tutorials/basic_training
폴더 명 : 2024_synthetic_biometric\fingerprint\Diffusion

![image](https://github.com/user-attachments/assets/0c30b273-cede-4f5a-a8e2-cf7bcbaed95c)

**2. Ridgepattern-GAN**
학습 : python train.py --modality fingerprint --net_name R_Enhancement --data_dir ../../dataset/KISA_DB_train/train_fingerprint/ --exp_name fingerprint_R_Enhancement --save_epochs 20 --epochs 2000 --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 64
![image](https://github.com/user-attachments/assets/fa4ad1e8-1afe-49bc-bd78-e9424c023d76)

**3. ID-preserving fingerprint(Private Code)**
   참고 : fingerprint\D-fingerprint(Binh)\_2019.03.27_Demo (2)\_2019.03.27_Demo (2)
   샘플 생성 파일(D-fingerprint weight file은 checkpoint 폴더에 저장)
   ![image](https://github.com/user-attachments/assets/40c9c1e3-0b40-40ee-85ff-7bd401e70ede)

demo_main_Diffusion.py : diffusion + ridgepattern 샘플 생성 이후 D-network 를 통한 서로 다른 ID 샘플 생성(score 비교하며 생성)

Demo_main_same_ID_with_log.py : D-network를 통한 샘플을 이용해 동일 ID 이지만 다른 지문 5장 생성, log file 자동 생성

![image](https://github.com/user-attachments/assets/661c1dbb-49f9-4819-8d14-55e97a02d5be)

**4. Style transfer(Wet,dry)**
2024_synthetic_biometric\fingerprint\image_processing\ridge_thicker_wet_dry.py
![image](https://github.com/user-attachments/assets/b0b8630e-e055-4d27-9480-3d905494cad4)
