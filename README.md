# DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration
### 데이콘 경진대회 - 이미지 색상화 및 손실 부분 복원 AI 알고리즘 개발
- - -
### 📚개요

데이콘에서 이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발을 주제로 하는 경진대회가 개최되어 참여해보았습니다.

경진대회에 사용되는 데이터를 소개하고 사용 모델과 외 기타 특징, 마지막으로 추후 개선 사항에 대한 설명으로 마무리하도록 하겠습니다. 

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
### 2. 사용 라이브러리

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=Numpy&logoColor=white">

<br/>

- - -

### 3. 모델 아키텍쳐(model architecture)

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

### 4. 손실 함수(Loss Function)

세 가지 손실 함수 픽셀 손실(Pixel Loss), SSIM 손실(Structural Similarity Index Loss), 부드러움 손실(Smoothness Loss)을 결합했습니다.

Pixel Loss에서 생성된 이미지와 원본 이미지 사이의 픽셀 단위 평균 제곱 오차(MSE)를 계산하고 픽셀 값 차이를 측정하여 생성된 이미지의 정확성을 평가합니다. SSIM Loss에서는 이미지의 전체적인 구조와 패턴 유사성을 고려하여 평가하며, Smoothness Loss로 이미지를 평활화하고 이미지들 간의 MSE를 계산합니다.

최종적으로 각 손실에 가중치(α, β, γ)를 적용해 다양한 측면으로 최적화합니다.

<br/>

![image](https://github.com/user-attachments/assets/ccd193df-30cf-402d-8ce1-1148288b012e)
> Loss Function

<br/>

- - -

### 5. 훈련 프로세스(training process)

앞서 설명했듯이 생성자와 판별자를 번갈아가며 훈련합니다. 

<br/>

![image](https://github.com/user-attachments/assets/21cfa54b-cb3b-4f5b-a22e-2715de0712e6)
> training process

<br/>

- - -

### 6. 기타(Other)

+ 가우시안 필터(Gaussian-Filter)

  가우시안 필터를 적용했습니다. 이미지의 노이즈를 제거하고 에지(Edge)를 부드럽게 만들어주는 효과가 있으며 Smoothness Loss를 평가하는 데 사용됩니다.

+ 에포크마다 테스트 이미지 생성 및 ZIP 파일로 저장

+ 체크포인트 저장 및 복원 기능

<br/>

- - -

### 7. 추후 개선 사항(improvements)

ReduceLROnPlateau와 같은 학습 후반부의 성능 향상을 기대할만한 LR scheduler를 도입하고 싶었지만 시간 제약으로 인해 하지 못했던 것이 많이 아쉬웠습니다. 이런 데이콘 경진대회가 또 개최된다면 배우는 마음으로 열심히 임해볼 생각입니다.

지금까지 긴 글 읽어주셔서 감사합니다.

<br/>

- - -
