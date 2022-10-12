# Project : CDK5 inhibitor 후보물질 발굴 및 설계

### 1. ligand_based_VS.py : torch_geometric.nn.data.Data 자료형으로 바꿔주는 전처리 과정을 수행
### 2. ligand_based_VS_2.py : 
     Generating node-level feature / edge type

     - Using only atom type (13 dim)
     - Each atom is embedded into 256 dimensional vector by a simple linear transformation
     - There are 6 edge type : the number of combinations of (bond type, bond stereo)
     
     Relational Graph Convolutional Network (RGCN)

     - Total 8 RGCN layers each of which has 256 channels
     - Skip-connections
     - Using the sum of node representations followed by one hidden layer MLP as the graph representation
     
     Readout phase

     - Multi layers perceptron with 2 hidden layers (1,024 dim, 512 dim)
     - Dropout with p=0.3
     - Directly predicting ST1 gap
     
     10-Fold ensembling

     - Taking the simple average of 10 models
     reference: https://github.com/HiddenBeginner/samsung-ai-challenge
     
 ### 3. ligand_crawling.py : zinc15에서 csv에 포함된 compounds의 sdf파일을 다운로드 받아줌
 ### 4. molecule_design.ipynb : 생명정보학회 현장중심 AI신약개발교육 22.9.30. 이경열님 강의자료 참고
 

## Project Overview
![CDK5 inhibitor 후보물질 발굴 및 설계](https://user-images.githubusercontent.com/96029849/195245083-bd410b0b-7474-485f-a0ea-a023da1ef817.png)


![image](https://user-images.githubusercontent.com/96029849/195098507-65a18bb4-9f30-43d8-9817-e3ef4c6cbf57.png)
ChEMBL database


### 1.ChEMBL을 통해 CDK5 관련 화합물 2319종의 data를 수집한 뒤, Standard Type이 IC50이고 pChEMBL Value data가 있는 화합물 1393종을 Binding affinity 예측 모델의 훈련데이터로 사용했다.

### 2.RGCN

ligand_based_VS.py : torch_geometric.nn.data.Data 자료형으로 바꿔주는 전처리 과정을 수행
ligand_based_VS_2.py : 
     Generating node-level feature / edge type

     - Using only atom type (13 dim)
     - Each atom is embedded into 256 dimensional vector by a simple linear transformation
     - There are 6 edge type : the number of combinations of (bond type, bond stereo)
     
     Relational Graph Convolutional Network (RGCN)

     - Total 8 RGCN layers each of which has 256 channels
     - Skip-connections
     - Using the sum of node representations followed by one hidden layer MLP as the graph representation
     
     Readout phase

     - Multi layers perceptron with 2 hidden layers (1,024 dim, 512 dim)
     - Dropout with p=0.3
     - Directly predicting ST1 gap
     
     10-Fold ensembling

     - Taking the simple average of 10 models
 
 reference: https://github.com/HiddenBeginner/samsung-ai-challenge

![image](https://user-images.githubusercontent.com/96029849/195098726-a4bd9207-f132-42b1-83f8-7b4c029236e6.png)
ZINC15 database

### 3.ZINC Database에서 Drug likeness, Synthetic Accessibility를 고려하여 LogP, MW와 perchasability를 filter로 화합물 data를 수집하였고 그 중 100,000종을 이번 project에 사용하였다.

### 4.Ligand-Based virtual screening
학습시킨 RGCN으로 ZINC Database에서 수집한 100,000종의 화합물의 pIC50을 예측하여 pIC50이 높은 100개의 화합물을 선별하였다.

### 5.Structure-based virtual screening

![image](https://user-images.githubusercontent.com/96029849/195099325-3bf18826-ae98-4670-9198-eaede03819d0.png), ![image](https://user-images.githubusercontent.com/96029849/195099381-92df06c4-aa80-486d-975e-915fa0fb8160.png)

PDB에서 CDK5단백질의 구조를 PDB 파일로 다운로드 받고 UCSF Chimera 프로그램으로 이미 알려진 리간드와 결합 region을 확인하였다.
1차로 선별된 100개의 화합물에 대해 Zinc15 데이터베이스에서 sdf파일을 다운받은 뒤 pdb 파일로 변환, 최종적으로 pbdqt파일로 변환한 뒤
linux환경에서 AutoDock Vina로 100종의 화합물을 대상으로 Docking score를 계산하였고 Docking Score가 높은 순으로 20종을 선별하였다.

선별된 20종을 대상으로 QED score, Novelty, 구조를 고려해서 10종을 최종 선정하였다.

### 6.De Novo Design

선정된 10종의 화합물을 기본 골격으로 De Novo Design을 수행하였습니다.
369개의 Building block과 64개의 반응식을 통해 최대 5번의 반응을 simulation하여 QSAR score이 개선된 화합물을 확인하였습니다. QSAR score 예측은 Random Forest Regression을 사용하여 예측하였습니다.
QSAR 개선만 고려하였을 때 결과물의 구조를 확인하였을 때, CDK5의 기존 신약후보물질(dinaciclib, selicicilib 등)과 구조적 유사성이 매우 적고 Drug likeness, Synthetic Accessibility도 부적절한 것 같아 QED score 개선 또한 기준으로 하여 De novo design을 수행하였고 기준에 부합하는 1종의 화합물을 얻었습니다.

 ![image](https://user-images.githubusercontent.com/96029849/195099850-1bf908e2-83f7-44d8-8465-a2fdb23a99c6.png)
### 선정된 화합물을 바탕으로 optimization한 구조
