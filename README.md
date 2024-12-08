# DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration
### 데이콘 경진대회 - 이미지 색상화 및 손실 부분 복원 AI 알고리즘 개발
- - -
### 📚개요

데이콘에서 이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발을 주제로 하는 경진대회가 개최되어 참여해보았습니다.

경진대회에 사용되는 데이터를 소개하고 사용 모델과 기타 추가 개선 사항, 마지막으로 실험 결과를 비교하며 마무리하도록 하겠습니다. 

<br/>

- - -

### 데이터(dataset info)

+ __train_input[폴더]__ : 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00329.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00555.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TRAIN_00650.png" width="200" height="200"/>

<br/>

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

+ __sample_submission.zip[제출양식]__ : 추론한 PNG이미지들을 zip 형식으로 압축한 제출 양식

<br/>

- - -
### 사용 라이브러리

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=Numpy&logoColor=white">

<br/>

- - -
### 하이퍼 파라미터(hyper parameters)

에포크마다 테스트 이미지 생성 및 ZIP 파일로 저장되게 설정했기 때문에 100으로 설정했습니다.

<br/>

```ruby
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

train_dataset = ImageDataset("train_input", "train_gt")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

epochs = 100
```
> hyper parameters

<br/>

- - -

### 데이터 전처리(Data Preprocessing)

```cv2.imread()```를 사용하여 입력 이미지와 정답 이미지를 로드하고 이 이미지들을 ```torch.tensor```로 변환 후 ```(C, H, W)```로 재배열했습니다.

그 후, 픽셀 값을 ```[0, 1]```로 정규화했습니다.

<br/>

```ruby
class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )

```
> Data Preprocessing

<br/>

- - -

### 모델 아키텍쳐(model architecture)

모델은 U-Net과 PatchGAN을 결합해 사용했습니다.

기본 GAN 구조는 생성자(Generator)가 랜덤 노이즈를 입력받아 이미지를 생성하고 판별자(Discriminator)가 진짜 또는 가짜 이미지인지 구분하는 구조지만 부분 이미지 및 복원에 특화된 PatchGAN을 결합했고,

실제 동작 과정은 U-Net에서 입력 이미지를 기반으로 복원/변환 이미지를 생성하고 PatchGAN은 그대로 생성된 이미지와 원본 이미지를 구분합니다.

두 개를 결합해 사용한 이유는 부분 영역을 디테일하게 복원하는 효과를 기대하며 두 개를 결합했습니다.

<br/>

```ruby
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)
        # ... (다른 디코더 블록들)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

```
> U-Net Generator

<br/>

아래 PatchGAN은 생성된 이미지 전체가 아닌 이미지의 작은 패치들을 판별합니다.

```ruby
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # ... (다중 합성곱 레이어)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
```
> PatchGAN Discriminator

<br/>


- - -

### 훈련 프로세스(training process)

앞서 설명했듯이 생성자와 판별자를 존재하고 번갈아가며 훈련합니다.

생성자는 더 자연스러운 이미지를 생성하도록 학습하고, 판별자는 진짜 이미지와 가짜 이미지를 구분하도록 학습합니다.

<br/>

```ruby
for epoch in range(epochs):
    for input_images, gt_images in train_loader:
        # 생성자 훈련
        fake_images = generator(input_images)
        g_loss_adv = adversarial_loss(discriminator(fake_images), real_labels)
        g_loss_pixel = pixel_loss(fake_images, gt_images)
        
        # 판별자 훈련
        pred_real = discriminator(gt_images)
        pred_fake = discriminator(fake_images.detach())
        d_loss = (adversarial_loss(pred_real, real_labels) + 
                  adversarial_loss(pred_fake, fake_labels)) / 2
```
> training process

<br/>

- - -

### 기타 및 개선 사항(Other improvements)

+ __가우시안 필터(Gaussian Filter)__

  가우시안 필터를 적용했습니다. 가우시안 필터는 이미지의 노이즈를 제거하고 에지(Edge)를 더 부드럽게 만들어주기에 이미지의 전체적인 자연스러운 느낌을 살리기 위해 사용했습니다.

<br/>

```ruby
def gaussian_filter(x, kernel_size=5, sigma=1.0):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
        torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(x.size(1), 1, 1, 1)

    padded_x = F.pad(x, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    filtered = F.conv2d(padded_x, gaussian_kernel, groups=x.size(1))
    
    return filtered
```
> Gaussian Filter

<br/>

+ __손실 함수(Loss Function)__

  세 가지 손실 함수 픽셀 손실(Pixel Loss), SSIM 손실(Structural Similarity Index Loss), 부드러움 손실(Smoothness Loss)을 결합했습니다.

  Pixel Loss에서 생성된 이미지와 원본 이미지 사이의 픽셀 단위 평균 제곱 오차(MSE)를 계산하고 픽셀 값 차이를 측정하여 생성된 이미지의 
  정확성을 평가합니다. SSIM Loss에서는 이미지의 전체적인 구조와 패턴 유사성을 고려하여 평가하며, Smoothness Loss로 이미지를 평활화
  하고 이미지들 간의 MSE를 계산합니다.

  최종적으로 각 손실에 가중치(α, β, γ)를 적용해 다양한 측면으로 최적화합니다.

  가우시안 필터와 마찬가지로 이미지의 전체적인 자연스러운 느낌을 살리기 위해 사용했으며, 단일 손실 함수의 한계를 극복하기 위해 사용했습니다.

<br/>

  ```ruby
  class CustomLoss(nn.Module):
    def forward(self, pred, target):
        pixel_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        
        pred_smooth = gaussian_filter(pred)
        target_smooth = gaussian_filter(target)
        smoothness_loss = F.mse_loss(pred_smooth, target_smooth)
        
        total_loss = (self.alpha * pixel_loss + 
                      self.beta * ssim_loss + 
                      self.gamma * smoothness_loss)
  ```
  > Loss Function

<br/>

- - -

### 실험 결과(Experiment results)

|**Model**|**Add parameter**|**Epoch**|**discriminator_loss**|**generator_loss**|**Dacon Score**|
|:------:|:---:|:---:|:---:|:---:|:---:|
|U-Net + PatchGAN||10|1.251|1.425|0.4642|
|U-Net + PatchGAN||20|1.071|1.225|0.4772|
|U-Net + PatchGAN||50|0.735|0.884|0.5071|
|U-Net + PatchGAN|+ Gaussian Filter|10|1.214|1.382|0.4680|
|U-Net + PatchGAN|+ Gaussian Filter|20|1.084|1.191|0.4772|
|U-Net + PatchGAN|+ Gaussian Filter|30|0.781|0.890|0.5115|
|U-Net + PatchGAN|+ Gaussian Filter|50|테스트2|테스트3|테스트3|

<br/>

+ __U-Net + PatchGAN(Epoch=10)__

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_010_Epoch10.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_035_Epoch10.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_054_Epoch10.png" width="200" height="200"/>

<br/>

+ __U-Net + PatchGAN(Epoch=50)__

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_010_Epoch50.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_035_Epoch50.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_054_Epoch50.png" width="200" height="200"/>

<br/>

+ __U-Net + PatchGAN + Gaussian Filter(Epoch=10)__

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_010_GF_Epoch10.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_035_GF_Epoch10.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_054_GF_Epoch10.png" width="200" height="200"/>

<br/>

+ __U-Net + PatchGAN + Gaussian Filter(Epoch=50)__

<img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_010_GF_Epoch50.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_035_GF_Epoch50.png" width="200" height="200"/> <img src="https://github.com/ShinBangHo/DACON-Development_of_AI_algorithm_for_image_colorization_and_loss_restoration/blob/main/TEST_054_GF_Epoch50.png" width="200" height="200"/>

<br/>

- - -

### 추후 개선 사항(improvements)

ReduceLROnPlateau와 같은 학습 후반부의 성능 향상을 기대할만한 LR scheduler를 도입하고 싶었지만 시간 제약으로 인해 하지 못했던 것이 많이 아쉬웠습니다. 이런 데이콘 경진대회가 또 개최된다면 배우는 마음으로 열심히 임해볼 생각입니다.

지금까지 긴 글 읽어주셔서 감사합니다.

<br/>

- - -
