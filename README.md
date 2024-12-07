# DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration
### 데이콘 경진대회 - 이미지 색상화 및 손실 부분 복원 AI 알고리즘 개발
- - -
### 개요

데이콘에서 이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발을 주제로 하는 경진대회가 개최되어 참여해보았습니다.

경진대회에 사용되는 데이터를 소개하고 사용 모델에 대한 설명, 마지막으로 추후 개선할 점과 느낀 점을 말하며 마무리하도록 하겠습니다. 

<br/>

- - -

### 1. 데이터(dataset info)

+ __train_input[폴더]__ : 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00329.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00555.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00650.png" width="200" height="200"/>
<br/><br/>

+ __train_gt[폴더]__ : 원본 PNG 이미지 (target, 29603장)

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00329%20(1).png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00555%20(1).png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00650%20(1).png" width="200" height="200"/> 

<br/>

+ __train.csv[파일]__ : 학습을 위한 Pair한 PNG 이미지들의 경로

<br/>

+ __test_input[폴더]__ : 흑백, 일부 손상된 PNG 평가 이미지 (input, 100장)

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_094.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_064.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_069.png" width="200" height="200"/>

<br/>

+ __test.csv[파일]__ : 추론을 위한 Input PNG 이미지들의 경로

<br/>

+ __sample_submission.zip[제출양식]__

  >추론한 PNG이미지들을 zip 형식으로 압축한 제출 양식

<br/>

- - -

### 2. 모델 아키텍쳐(model architecture)

모델은 U-Net과 PatchGAN을 결합해 사용하였습니다.

기본 GAN 구조는 생성자(Generator)가 랜덤 노이즈를 입력받아 이미지를 생성하고 판별자(Discriminator)가 진짜 또는 가짜 이미지인지 구분하는 구조였지만 부분 이미지 및 복원에 특화시키기 위해 U-Net과 PatchGAN을 결합하였습니다.

실제 동작 과정은 U-Net에서 입력 이미지를 기반으로 복원/변환 이미지를 생성하고 PatchGAN은 그대로 생성된 이미지와 원본 이미지를 구분합니다.

<br/>

![image](https://github.com/user-attachments/assets/6728bcf0-31f3-4830-a827-c80de04e0615)
> U-Net Generator

<br/>

아래 PatchGAN은 생성된 이미지 전체가 아닌 이미지의 작은 패치들을 판별합니다.

![image](https://github.com/user-attachments/assets/64a32908-616f-4e24-9c12-a5e3b68042a1)
> PatchGAN Discriminator

<br/>

- - -

### 2. 손실 함수(Loss Function)

손실 함수를 사용하여 이미지 복원 개선
