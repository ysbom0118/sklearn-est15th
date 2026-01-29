# 🤖 Scikit-learn Machine Learning Hub (EST 15th)

이 저장소는 **EST 15기** 과정 중 진행된 Scikit-learn 기반의 머신러닝 학습 소스코드와 프로젝트 결과물을 담고 있습니다. 데이터 전처리부터 모델링, 하이퍼파라미터 튜닝, 그리고 프로젝트 실습까지의 전 과정을 포함합니다.

---

## 📂 주요 목차 (Contents)

### 1. 머신러닝 기초 (ML Fundamentals)
- `1_sklearn_start.ipynb`: Scikit-learn 시작하기 및 기본 구조 이해
- `2_ModelSelection.ipynb`: 학습/테스트 데이터 분리 및 교차 검증
- `4_sklearn_PreProcess.ipynb`: 데이터 인코딩, 스케일링 등 전처리 가이드

### 2. 분류 모델 (Classification)
- `3_SVM.ipynb`: 서포트 벡터 머신(SVM)의 이해와 실습
- `5_sklearn_classification.ipynb`: 주요 분류 알고리즘 (Decision Tree, Random Forest 등)
- `6_classification_Optuna.ipynb`: Optuna를 이용한 분류 모델 하이퍼파라미터 최적화

### 3. 회귀 모델 (Regression)
- `9_LinearRegressionModel.ipynb`: 선형 회귀의 기본과 응용
- `8_polynominal_Feature.ipynb`: 다항 회귀와 피처 엔지니어링
- `Plus_6_LinearRegressionModel.ipynb`: 회귀 모델 심화 실습

### 4. 실전 데이터 분석 프로젝트 (Projects)
- **🚢 Titanic Survivor Prediction**
  - `7_Titanic.ipynb`: 타이타닉 생존자 예측 기본 모델링
  - `타이타닉생존자예측-데이터전처리.ipynb`: 상세 데이터 정제 및 피처 엔지니어링
  - `20260124_과제_***_titanic.ipynb`: 개인 과제 및 최종 분석 리포트
- **🍷 Wine Quality Analysis**
  - `Plus_1_sklearn_wine_classification.ipynb`: 와인 종류 분류
  - `Plus_2_Red_wine_quality_analysis_회귀로solve.ipynb`: 와인 품질 점수 예측 (회귀)
- **🔢 Digits & Other Datasets**
  - `Plus_3_sklearn_disits.ipynb`: 손글씨 숫자(Digits) 분류 실습

### 5. 앙상블 및 고급 기법 (Advanced Topics)
- `10_ensemble.ipynb`: 보팅(Voting), 배깅(Bagging), 부스팅(Boosting) 기초
- `Plus_7_ensemble.ipynb`: 다양한 앙상블 기법 최적화 및 비교
- `11_ensemble_Optuna.ipynb`: 앙상블 모델의 하이퍼파라미터 튜닝
- `13_unsupervisedLearning.ipynb`: 비지도 학습 (Clustering, PCA 등)
- `AutoML/`: 자동화 머신러닝 활용 예제

### 6. 데이터 시각화 (Visualization)
- `folium_visualization_colored.ipynb`: Folium을 이용한 지도 기반 데이터 시각화
- `캘리포니아집값_시각화_folium_visualization_colored.ipynb`: 캘리포니아 주택 가격 데이터 지형 시각화

---

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Libraries**:
  - `scikit-learn`: 주요 머신러닝 알고리즘 및 도구
  - `pandas`, `numpy`: 데이터 조작 및 수치 계산
  - `matplotlib`, `seaborn`, `folium`: 데이터 시각화
  - `optuna`: 하이퍼파라미터 튜닝 최적화 framework
  - `autogluon`: Auto ML 활용 (일부 프로젝트)

---

## 🚀 시작하기 (How to Use)

1. **저장소 클론**
   ```bash
   git clone https://github.com/ysbom0118/sklearn-est15th.git
   ```

2. **환경 설정 (Python 가상환경 권장)**
   ```bash
   pip install -r requirements.txt  # (필요 시 패키지 개별 설치)
   ```

3. **주피터 노트북 실행**
   ```bash
   jupyter notebook
   ```

---

## ✍️ Author
- **Name**: KBM (EST 15기)
- **GitHub**: [@ysbom0118](https://github.com/ysbom0118)

---
*이 저장소는 꾸준한 학습과 기록을 위해 관리되고 있습니다.*
