# DACON-Image_Colorization_and_Missing_Parts_AI_Competition
### 데이콘 경진대회 - 이미지 색상화 및 손실 부분 복원 AI 알고리즘 개발
- - -
### 개요

데이콘에서 이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발을 주제로 하는 경진대회가 개최되어 참여해보았습니다.

경진대회에 사용되는 데이터를 소개하고 코드에 대한 설명, 마지막으로 추후 개선할 점과 느낀 점을 말하며 마무리하도록 하겠습니다. 

- - -

### 1. 데이터(dataset info)

__train_input[폴더]__ : 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)
__train_gt[폴더]__ : 원본 PNG 이미지 (target, 29603장)
__train.csv[파일]__ : 학습을 위한 Pair한 PNG 이미지들의 경로


__test_input[폴더]__ : 흑백, 일부 손상된 PNG 평가 이미지 (input, 100장)
__test.csv[파일]__ : 추론을 위한 Input PNG 이미지들의 경로
