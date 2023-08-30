# Capston

**전이학습을 통한 백반증 영역 검출**

**및 병변 심각도 측정**

**김성훈, 박소연, 홍석표**

**1. 배경 및 필요성**
**
` `백반증은 피부의 멜라닌 색소가 소실되어 피부의 여러 형태의 흰색 반점이 생기는 난치성 피부질환이다. 백반증은 난치성 피부질환 중에는 흔하게 나타나는데 전체 인구 중 0.5~2% 정도에게 나타나며 인종 지역과 차이 없이 발생한다. 백반증은 통증도 없고 전염성도 없는, 걸리게 되더라도 건강에는 문제가 없는 피부질환이다. 하지만 밖으로 노출되는 부위에 백반증이 생긴다면 하얀 반점이 흉하게 느껴져 사람을 대할 때 위축감을 느끼는 등 정상적인 사회활동이 어려워질 수 있다.[1] 따라서, 백반증의 진행정도가 외부에 티가 나게 되는 정도가 되면 피부과를 방문하게 되는데, 진행이 많이 된 경우 치료하기가 어렵고 초기에 치료를 해야 병의 진행을 막고 호전시키기 쉽다.[2] 그러므로 백반증을 판별하기 위한 백반증 환부 영역검출이 중요하다. 일반적으로는 육안으로 백반증을 판별하거나 피부과에서 우드등(Wood's lamp)을 통해 발병 부위를 확인하지만 병 초기에는 피부과를 방문할 정도로 심각하게 여기지 않을 가능성이 크고 육안으로는 진행도 판단의 기준이 명확하지 않을 가능성이 있다.[3] 따라서 이에 도움을 주고자, 딥러닝을 통한 백반증 영역 검출방법을 제안해보고자 한다.

백반증 환부의 영역검출을 위해 다양한 방법으로 연구한 논문이 다수 있다. 하지만 이러한 논문들은 영역 검출까지만 진행되었고 그 이상까지 진행되지는 않았다. 본 연구에서는 딥러닝을 통해 백반증 발병 부위를 검출한 후, 발병부위와 비발병부위의 색상 차이를 계산하여 백반증 병변 진행도까지 확인하고자 한다.

**2. 관련 연구**

**2-1. 이미지 처리**

국내에서는 이미지 처리 기법을 통한 백반증 영역 검출에 대한 연구만 다수 진행되어 왔다. 백반증 발병 영역을 일정 수준으로 검출할 순 있었으나, 백반증이 아닌 영역까지 검출되어 알고리즘에 추가 보완이 필요한 상태이다.[4] 

해외 연구에서는, 국내 연구와는 다르게 Compact, Quickshift, Slic, Felzenszwalb, Watershed, RAG 등의 다양한 영상 분할 기술을 사용하였지만, 이는 성능을 비교분석한 연구이며 명확한 영역검출을 위한 알고리즘을 파악하기에는 한계가 있다. [5]

**2-2. Semantic segmentation**

우선, 컴퓨터 비전에서 가장 많이 다뤄지는 문제들은 다음과 같은 것들이 있다.

Classification (분류)는 인풋에 대해서 하나의 물체를 구분하는 작업을 말한다. Object Detection (객체 탐지)는 물체를 구분함과 동시에, 그 물체가 어디에 있는지까지 Boxing하는 작업이다. 나머지 하나는 Segmentation (분할)인데, 모든 픽셀에 대해, 각 픽셀이 어떤 물체의 class인지 구분하는 작업이다.
**
` `이중 Segmentation에는** 대표적으로 U-Net을 활용한 방법으로 의료용 이미지에 대한 전이학습을 진행한 연구가 있다. [6]

` `본 논문에서는 FCN 기반모델과 Deeplab 기반모델를 활용한 전이학습을 통한 병변 영역검출 방법을 제안한다.

**3.** **Method**

**3-1. 데이터셋**

학습을 위한 데이터셋은 구글링을 통해 약 425개의 백반증 환부를 확대한 이미지를 수집하였다. 수집한 뒤, 백반증이 진행되고 있는 환부의 영역을 선택해 따로 추출하기 위한 라벨링을 거쳐 학습을 진행하였다. Training에 300개의 이미지가 사용되었고 Validation에 125개의 이미지가 사용되었다. 의료데이터는 사적인 부분 때문에 이미 라벨링을 거친 공개적인 자료를 구하는데 한계가 있었다. 따라서 직접 이미지 라벨링을 수행하는 방향으로 결정하였다. 

학습을 시작하기 전에, 이미지에 대한 데이터 전처리를 진행하였다. 312 x 312 사이즈로 resizing한 뒤, 256 x 256으로 랜덤하게 자르는 방법을 택했다. 이후 50%확률로 좌우 반전을 수행한 후, 50%확률로 상하 반전을 수행하였다.

밝기도 마찬가지로 50% 확률로 수행하고 0.7 ~ 1.3 사이의 난수를 통해 밝기 조절을 하였다. 

학습을 할 때, 설정해 놓은 데이터 전처리 방법을 사용하여 데이터를 가져오게 된다. 

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.001.jpeg)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.002.png)

**Fig1.  Vitiligo Dataset**

**3-2. 전이학습**

전이학습 이란, 임의의 영역에서 학습된 신경망 네트워크의 일부를 다른 영역에 적용하여 모델을 학습하는 방법이다.[7] 높은 정확도를 비교적 짧은 시간 내에 만들어 낼 수 있기 때문에, 딥러닝에서 유망한 방법론 중 하나이다. 즉, 사전학습 된 모델을 이용하는 것을 뜻한다. 

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.003.jpeg)

**Fig2. Transfer learning**

전이학습의 가장 큰 장점은 학습시간을 크게 줄일 수 있다는 점이다. 해결해야하는 것과 유사한 유형의 문제를 처리한 신경망이 있는지 사전에 탐색해보고, 그 신경망의 일부를 재사용함으로써 훈련속도를 높일 수 있다.[8] 또한, 전이학습을 사용함으로써 데이터셋의 양이 충분하지 않더라도 우수한 학습효과를 얻을 수 있다. 일정수준 이상의 학습효과를 얻으려면 대규모의 데이터셋이 필요한데, 의료관련 데이터셋과 같이 일반인이 얻기 힘든 데이터셋은 충분한 양을 확보하는데 어려움이 있다. 이러한 경우 전이학습을 적용하면, 비슷한 유형의 대규모 데이터를 통해 시스템을 학습시킨 후, 주어진 소수의 데이터셋을 적용하여 기존의 신경망을 사용한 재학습과정인 fine – tuning을 하면 우수한 정확도를 도출할 수 있다.[9] 

본 논문에서는 의료데이터인 백반증 환부 데이터셋을 대규모로 확보하기 어려운 점을 고려하여, 전이학습을 적용하여 학습을 진행하였다. 

**3-3. 모델**

Segmentation 관련 모델에는 대표적으로 크게 U-Net, DeepLab, FCN 이렇게 3가지가 있다. [10]

U-Net은 생의학 영상분할을 위해 설계된 모델로, U자형의 인코더-디코더 구조를 가지고 있다.[11]

Deeplab은 합성곱 층의 앞부분은 기존 CNN과 같은 커널을 사용하고 확장 합성곱, 공간 피라미드 풀링 등의 기법이 사용되었다. [12]

FCN(Fully Convolutional Network) 기반 모델은 fully connected 층을 제거하고 모든 층을 합성곱 층(convolutional layer)으로만 구성된다. [13] 특징점 추출의 결과는 Fig3와 같이 특징점 추출의 결과가 픽셀 별로 미리 정의된 몇 개의 클래스에 대한 확률 지도(probability map)형태로 나온다. [14]

FCN과 Deeplab의 구조는 합성곱 층을 통해 얻은 특징점 지도를 원본 이미지 크기와 같게 만들기 위해서 단순하게 up-sampling 하는 구조를 사용한다. 

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.004.png)

**Fig3.** **Network structure for classification and feature point extraction network structure of FCN [7]**

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.005.jpeg)

**Fig4.  FCN Architecture**

` `본 연구에서는 FCN기반 모델, Deeplab기반 모델 몇개를 비교해서 높은 결과값을 보여주는 모델을 채택 예정이다. 

**3-4. 성능 평가**

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.006.png)

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.007.png)

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.008.png)

Acurracy는 데이터 분포에 민감하다. 따라서 imbalanced한 데이터에 사용하기에는 적합하지 않다.

그런데, Precision 과 Recall은 imbalanced한 데이터를 사용한 학습 상황에서도 평가지표로 효과적으로 사용할 수 있다. F-Measure는 이러한 Precision 과 Recall을 β 계수로 가중치를 부여하여 합친 하나의 평가지표이다. f1 – score은 F-Measure의 β를 1로 정하여 Precision 과 Recall에 가중치를 동일하게 부여한 평가지표로 본 연구에서 사용한 imbalanced한 데이터에 일반적으로 사용된다. [15]

본 연구에서 사용한 데이터셋의 이미지 중

imbalanced한 데이터가 존재한다. 

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.009.png)

**Fig5. Dataset**

위의 이미지는 전체 픽셀 수 65536(256 x 256) 개 중 0 라벨이 62192개, 255 라벨이 3344개로 백반증 환부가 아닌 영역이 약 95%로 데이터의 라벨이 치우쳐 있다. 따라서, 본 모델의 성능평가를 위해 f1-score를 사용하기로 결정하였다. 






**4. 결과**


<table><tr><th colspan="1" valign="top"><b>기반 모델</b></th><th colspan="1" valign="top"><b>수축단</b></th><th colspan="1" valign="top"><b>F1-score</b></th></tr>
<tr><td colspan="1" rowspan="3" valign="top"><p></p><p>Deeplab</p></td><td colspan="1" valign="top">resnet101</td><td colspan="1" valign="top">86\.19</td></tr>
<tr><td colspan="1" valign="top">resnet50</td><td colspan="1" valign="top">86\.01</td></tr>
<tr><td colspan="1" valign="top">mobilenet_v3</td><td colspan="1" valign="top">83\.82</td></tr>
<tr><td colspan="1" rowspan="3" valign="top"><p></p><p>Unet</p></td><td colspan="1" valign="top">resnet101</td><td colspan="1" valign="top">85\.694</td></tr>
<tr><td colspan="1" valign="top">resnet50</td><td colspan="1" valign="top">85\.23</td></tr>
<tr><td colspan="1" valign="top">mobilenet_v3</td><td colspan="1" valign="top">83\.88</td></tr>
<tr><td colspan="1" rowspan="2" valign="top">FCN</td><td colspan="1" valign="top">resnet101</td><td colspan="1" valign="top">84\.51</td></tr>
<tr><td colspan="1" valign="top">resnet50</td><td colspan="1" valign="top">84\.79</td></tr>
</table>

각 모델마다 20epoch만큼 학습하였으며,

Deeplabv3\_resnet101의 F1-score가 가장 높게 나온 관계로 이 모델로 학습을 진행하였다. 

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.010.jpeg)

**Fig6. Deeplabv3\_resnet101 학습 결과**

![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.011.png)

**Fig7. 원본이미지, Ground truth, Prediction**

` `(Fig6) 원본이미지, Ground truth, prediction을 비교한 결과, 일정수준이상 비슷한 영역검출 결과를 보이는 것으로 확인되었다. 

백반증 병변진행도 확인을 위해, 학습된 모델을 통해 원본 이미지를 병변이미지와 비병변 이미지로 나누어서 두 이미지의 RGB값의 평균을 측정하였다. 임의로 백반증의 진행정도를 상, 중, 하 로 분류하여 각각의 이미지를 선정하였다.  

|![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.012.jpeg)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.013.png)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.014.png)|
| - |
|![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.015.jpeg)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.016.png)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.017.jpeg)|
|![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.018.jpeg)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.019.png)![](Aspose.Words.5f38b4e8-c2ea-40c8-bd67-015bd60460f0.020.jpeg)|

**Fig8. 백반증 진행도에 따른 각각의 원본이미지,**

` `**병변이미지, 비병변이미지 (상, 중, 하)**

각각의 사진은 아래와 같은 RGB값의 차이를 가지며 백반증의 진행 정도에 따라 병변과 비병변영역의 RGB 평균수치의 차이가 있었다.

||**병변 RGB 평균**|**비병변 RGB 평균**|**차이**|
| - | :-: | :-: | :-: |
|상|136\.0|91\.2|44\.8|
|중|127\.9|101\.7|26\.2|
|하|188\.0|171\.0|17\.0|

**5. 결론**

본 논문에서는 백반증 환부 영역을 FCN기반 Model과 Deeplab기반 모델을 활용하여segmentation을 수행한 뒤, 전이학습을 적용한 데이터셋 훈련과정을 통한 영역 검출과 예측된 마스크 이미지를 통해 병변이미지의 RGB값을 측정하여 그 차이를 비교하여 병변진행도 확인을 할 수 있었다. 

본 연구는 기존 백반증 환부 영역검출에 잘 쓰이지 않던 모델인 Deeplab기반 모델을 통한 영역 검출을 수행했다는 점에서 의미가 있다. 또한, 백반증 환부 이미지가 의료데이터이기 때문에, 데이터셋을 대량으로 확보하기 어렵다는 한계점을 전이학습을 적용해 해결했다는 점에서 주목할 만하다. 

그러나, 데이터셋 라벨링을 직접 수행한 결과, 예상했던 것보다 정확도가 떨어진 것을 확인할 수 있었다. 따라서 더 양질의 데이터셋을 활용할 필요가 있어 보인다. 

본 연구를 발전시켜, 추후에는 양질의 데이터셋과 병변진행도 확인을 위한 추가적인 알고리즘 보완을 통해 백반증 환자를 위한 치료일지 애플리케이션 개발까지 진행할 계획이다.  이는 백반증 진행도 확인에 실용적인 도움을 줄 것으로 기대한다.  

**6. 참고 문헌**

[1]	G. Hautmann and E. Panconesi, “Vitiligo: a psychologically influenced and influencing disease.,” *Clin. Dermatol.*, vol. 15, no. 6, pp. 879–890, 1997, doi: 10.1016/s0738-081x(97)00129-6.

[2]	C. W. Choi, “Non-surgical treatment of vitiligo,” *J. Korean Med. Assoc.*, vol. 63, no. 12, pp. 741–747, Dec. 2020, doi: 10.5124/jkma.2020.63.12.741.

[3]	S.-H. OH, “Classification and diagnosis of vitiligo,” *J. Korean Med. Assoc.*, pp. 731–740, 2020.

[4]	S.-W. Shin, K.-S. Kim, S.-M. Lee, and J.-H. Kim, “Color Image Segmentation of Vitiligo Region,” in *Proceedings of the KIEE Conference*, 2011, pp. 2037–2038.

[5]	N. Agrawal and S. Aurelia, “A Review on Segmentation of Vitiligo image,” *IOP Conf. Ser. Mater. Sci. Eng.*, vol. 1131, no. 1, p. 012003, Apr. 2021, doi: 10.1088/1757-899x/1131/1/012003.

[6]	D. Cheng and E. Y. Lam, “Transfer Learning U-Net Deep Learning for Lung Ultrasound Segmentation,” Oct. 2021, doi: 10.48550/arxiv.2110.02196.

[7]	N. Agarwal, A. Sondhi, K. Chopra, and G. Singh, “Transfer learning: Survey and classification,” in *Smart innovations in communication and computational sciences*, Springer, 2021, pp. 145–155.

[8]	S.-W. Park and D.-Y. Kim, “Comparison of image classification performance in convolutional neural network according to transfer learning,” *J. Korea Multimed. Soc.*, vol. 21, no. 12, pp. 1387–1395, 2018.

[9]	S. Bae Park, H.-K. Lee, and D. Sik Yoo, “An Efficient Guitar Chords Classification System Using Transfer Learning,” *J. Korea Multimed. Soc.*, vol. 21, no. 10, pp. 1195–1202, 2018, doi: 10.9717/KMMS.2018.21.10.1195.

[10]	I. Ahmed, M. Ahmad, F. A. Khan, and M. Asif, “Comparison of Deep-Learning-Based Segmentation Models: Using Top View Person Images,” *IEEE Access*, vol. 8, pp. 136361–136373, 2020, doi: 10.1109/ACCESS.2020.3011406.

[11]	W. Weng and X. Zhu, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” *IEEE Access*, vol. 9, pp. 16591–16603, May 2021, doi: 10.1109/ACCESS.2021.3053408.

[12]	L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 40, no. 4, pp. 834–848, 2017.

[13]	E. Shelhamer, J. Long, and T. Darrell, “Fully Convolutional Networks for Semantic Segmentation,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 4, pp. 640–651, 2017, doi: 10.1109/TPAMI.2016.2572683.

[14]	S.-Y. Park and Y.-S. Heo, “딥러닝을 이용한 Semantic Segmentation 기술 동향 분석,” 전기의세계, vol. 67, no. 7, pp. 18–24, 2018.

[15]	H. He and E. A. Garcia, “Learning from imbalanced data,” *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, 2009.

