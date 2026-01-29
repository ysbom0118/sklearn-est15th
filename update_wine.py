import json
import os

path = r'c:\Users\user\github\DataScience\scikit-learn\Plus_4_kaggle_red_wind.ipynb'
new_source = [
    '# Kaggle 레드 와인 품질(Red Wine Quality) 데이터 분석 및 예측\n',
    '\n',
    '이 프로젝트는 포르투갈의 \"Vinho Verde\" 레드 와인 샘플 데이터를 사용하여 와인의 화학적 특성이 품질에 미치는 영향을 분석하고, 이를 기반으로 와인 품질을 분류 및 예측하는 모델을 구축하는 데 목적이 있습니다.\n',
    '\n',
    '## 1. 데이터셋 개요\n',
    '- **출처**: [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)\n',
    '- **제공자**: Paulo Cortez (University of Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos, Jose Reis (CVRVV)\n',
    '- **데이터 구성**: 1,599개의 레드 와인 샘플과 12개의 변수 (11개의 독립 변수 + 1개의 종속 변수)\n',
    '\n',
    '## 2. 특성(Feature) 상세 설명\n',
    '\n',
    '### 입력 변수 (화학적 성질)\n',
    '1. **fixed acidity** (결합 산도): 와인의 비위발성 산성 성분. 맛과 보존성에 영향을 미칩니다.\n',
    '2. **volatile acidity** (휘발성 산도): 와인의 식초 향을 유발하는 아세트산 수치.\n',
    '3. **citric acid** (구연산): 와인에 신선함과 풍미를 더해주는 성분.\n',
    '4. **residual sugar** (잔류 당분): 발효 후 남은 설탕 양.\n',
    '5. **chlorides** (염화물): 와인에 포함된 소금의 양.\n',
    '6. **free sulfur dioxide** (유리 이산화황): 산화 방지를 위한 이산화황.\n',
    '7. **total sulfur dioxide** (총 이산화황): 모든 상태의 이산화황 수치.\n',
    '8. **density** (밀도): 와인의 밀도.\n',
    '9. **pH** (산성도): 와인의 산도.\n',
    '10. **sulphates** (황산염): 보존을 위한 첨가제.\n',
    '11. **alcohol** (알코올): 와인의 알코올 도수(%).\n',
    '\n',
    '### 출력 변수 (타겟)\n',
    '- **quality** (품질): 전문가 점수 (3 ~ 8 분포)\n',
    '\n',
    '## 3. 분석 및 예측 프로세스\n',
    '1. **EDA**: 데이터 분포 및 상관관계 분석\n',
    '2. **전처리**: 데이터 분할 및 스케일링\n',
    '3. **모델링**: 8가지 알고리즘 성능 비교\n',
    '4. **최적화**: 하이퍼파라미터 튜닝 및 앙상블 적용\n'
]

try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    nb['cells'][0]['source'] = new_source
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
