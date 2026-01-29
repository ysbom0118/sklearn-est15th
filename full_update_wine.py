import json
import os

path = r'c:\Users\user\github\DataScience\scikit-learn\Plus_4_kaggle_red_wind.ipynb'

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Kaggle 레드 와인 품질(Red Wine Quality) 데이터 분석 및 예측\n",
            "\n",
            "이 프로젝트는 포르투갈의 \"Vinho Verde\" 레드 와인 샘플 데이터를 사용하여 와인의 화학적 특성이 품질에 미치는 영향을 분석하고, 이를 기반으로 와인 품질을 분류 및 예측하는 모델을 구축하는 데 목적이 있습니다.\n",
            "\n",
            "## 1. 데이터셋 개요\n",
            "- **출처**: [Kaggle - Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)\n",
            "- **데이터 구성**: 1,599개의 레드 와인 샘플과 12개의 변수 (11개의 독립 변수 + 1개의 종속 변수)\n",
            "\n",
            "## 2. 특성(Feature) 상세 설명\n",
            "\n",
            "### 입력 변수 (화학적 성질)\n",
            "1. **fixed acidity** (결합 산도): 와인의 비휘발성 산성 성분 (주로 주석산). 와인의 맛과 보존성에 기여합니다.\n",
            "2. **volatile acidity** (휘발성 산도): 와인의 식초 향을 유발하는 아세트산 수치. 수치가 높으면 와인의 품질을 저하시킵니다.\n",
            "3. **citric acid** (구연산): 와인에 신선함과 풍미를 더해주며 산도를 조절합니다.\n",
            "4. **residual sugar** (잔류 당분): 발효 후 남은 설탕 양. 와인의 당도를 결정합니다.\n",
            "5. **chlorides** (염화물): 와인에 포함된 소금의 양.\n",
            "6. **free sulfur dioxide** (유리 이산화황): 산화와 미생물 번식을 방지하기 위해 첨가되는 가스 상태의 이산화황.\n",
            "7. **total sulfur dioxide** (총 이산화황): 유리 상태와 다른 성분에 결합된 이산화황의 총량.\n",
            "8. **density** (밀도): 설탕 함량과 알코올 농도에 따라 결정되는 와인의 무게감.\n",
            "9. **pH** (산성도): 와인의 산성 수준 (0: 강산성, 14: 강염기성). 대부분의 와인은 3~4 사이입니다.\n",
            "10. **sulphates** (황산염): 항균 및 항산화 효과를 강화하기 위해 첨가되는 성분.\n",
            "11. **alcohol** (알코올): 와인의 알코올 농도(%).\n",
            "\n",
            "### 출력 변수 (타겟)\n",
            "- **quality** (품질): 최소 3명의 와인 전문가가 매긴 0~10 사이의 점수 (본 데이터셋은 3~8 분포).\n",
            "\n",
            "## 3. 분석 및 예측 워크플로우\n",
            "1. **EDA**: 데이터 요약 및 시각화를 통한 인사이트 도출\n",
            "2. **전처리**: 이상치 검토 및 `StandardScaler`를 이용한 특성 스케일링\n",
            "3. **모델링**: 8가지 머신러닝 알고리즘 비교 테스트\n",
            "4. **최적화**: 상위 성능 모델 선정 및 `GridSearchCV` 튜닝\n",
            "5. **앙상블**: 소프트 보팅(Soft Voting) 기법을 통한 최종 예측기 구축"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import warnings\n",
            "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# 시각화 설정\n",
            "%matplotlib inline\n",
            "sns.set(style='whitegrid', palette='pastel')\n",
            "plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트 설정\n",
            "plt.rcParams['axes.unicode_minus'] = False\n",
            "\n",
            "# 데이터 로드\n",
            "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv\"\n",
            "df = pd.read_csv(url, sep=';')\n",
            "print(f\"데이터셋 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. 탐색적 데이터 분석 (EDA)\n",
            "데이터의 분포를 확인하고 특성 간의 관계를 파악합니다."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. 품질 분포 확인\n",
            "plt.figure(figsize=(8, 5))\n",
            "sns.countplot(data=df, x='quality')\n",
            "plt.title('와인 품질 분포')\n",
            "plt.show()\n",
            "\n",
            "# 2. 화학적 특성 분포 (히스토그램)\n",
            "df.drop('quality', axis=1).hist(figsize=(15, 12), bins=30, edgecolor='black')\n",
            "plt.suptitle('화학적 특성별 분포', fontsize=16)\n",
            "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. 특성 간 상관관계 히트맵\n",
            "plt.figure(figsize=(12, 10))\n",
            "mask = np.triu(np.ones_like(df.corr(), dtype=bool))\n",
            "sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)\n",
            "plt.title('특성 간 상관관계 히트맵')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. 데이터 전처리 및 특성 엔지니어링"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 데이터 분리\n",
            "X = df.drop('quality', axis=1)\n",
            "y = df['quality']\n",
            "\n",
            "# 학습/테스트 데이터 분할 (80:20)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "\n",
            "# 스케일링 (Standardization)\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print(\"데이터 전처리 완료.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. 8가지 모델 비교 분석 및 성능 시각화"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=1000),\n",
            "    'Random Forest': RandomForestClassifier(random_state=42),\n",
            "    'SVC': SVC(probability=True, random_state=42),\n",
            "    'K-Neighbors': KNeighborsClassifier(),\n",
            "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
            "    'Gradient Boosting': GradientBoostingClassifier(random_state=42),\n",
            "    'AdaBoost': AdaBoostClassifier(random_state=42),\n",
            "    'Extra Trees': ExtraTreesClassifier(random_state=42)\n",
            "}\n",
            "\n",
            "# 교차 검증을 통한 성능 비교\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)\n",
            "    results.append({'Model': name, 'Accuracy': cv_scores.mean()})\n",
            "\n",
            "results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)\n",
            "print(\"--- 모델 성능 순위 ---\")\n",
            "print(results_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 모델 성능 시각화\n",
            "plt.figure(figsize=(12, 6))\n",
            "sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')\n",
            "plt.title('8가지 모델 성능 비교 (교차 검증 정확도)')\n",
            "plt.xlim(0.4, 0.75) # 가독성을 위한 범위 조정\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. 상위 4개 모델 앙상블 및 최적화"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 상위 4개 모델 이름 추출\n",
            "top_4_names = results_df.head(4)['Model'].values\n",
            "print(f\"앙상블을 위해 선정된 상위 4개 모델: {top_4_names}\")\n",
            "\n",
            "# 하이퍼파라미터 튜닝 (범위 설정)\n",
            "param_grids = {\n",
            "    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},\n",
            "    'Extra Trees': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},\n",
            "    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.05]},\n",
            "    'SVC': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}\n",
            "}\n",
            "\n",
            "tuned_estimators = []\n",
            "for name in top_4_names:\n",
            "    if name in param_grids:\n",
            "        print(f\"{name} 튜닝 중...\")\n",
            "        grid = GridSearchCV(models[name], param_grids[name], cv=3, n_jobs=-1, scoring='accuracy')\n",
            "        grid.fit(X_train_scaled, y_train)\n",
            "        tuned_estimators.append((name, grid.best_estimator_))\n",
            "        print(f\"{name} 최적 파라미터: {grid.best_params_}\")\n",
            "    else:\n",
            "        # 튜닝 설정이 없는 경우 (예: LR 등 상위에 올라올 시)\n",
            "        m = models[name].fit(X_train_scaled, y_train)\n",
            "        tuned_estimators.append((name, m))\n",
            "\n",
            "# 보팅 앙상블 생성 (Soft Voting)\n",
            "ensemble_model = VotingClassifier(estimators=tuned_estimators, voting='soft')\n",
            "ensemble_model.fit(X_train_scaled, y_train)\n",
            "print(\"최종 앙상블 모델 구축 완료.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. 최종 결과 시각화 및 검증"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 성능 평가\n",
            "y_pred = ensemble_model.predict(X_test_scaled)\n",
            "final_acc = accuracy_score(y_test, y_pred)\n",
            "\n",
            "print(f\"최종 앙상블 모델 테스트 정확도: {final_acc:.4f}\")\n",
            "print(\"\\n--- 분류 보고서 ---\")\n",
            "print(classification_report(y_test, y_pred))\n",
            "\n",
            "# 혼동 행렬 시각화\n",
            "plt.figure(figsize=(10, 8))\n",
            "cm = confusion_matrix(y_test, y_pred)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
            "            xticklabels=np.unique(y), yticklabels=np.unique(y))\n",
            "plt.title('최종 앙상블 모델 혼동 행렬')\n",
            "plt.xlabel('예측')\n",
            "plt.ylabel('실제')\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(\"SUCCESS\")
