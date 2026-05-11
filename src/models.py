"""
models.py — Stage 3: Model Definitions (MULTICLASS — 6 defect types)
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from src.config import SVM_PARAMS, RF_PARAMS, GB_PARAMS


def build_svm() -> SVC:
    model = SVC(**SVM_PARAMS)
    print(f"  🔵 SVM created (kernel={SVM_PARAMS['kernel']}, C={SVM_PARAMS['C']}, "
          f"multiclass=ovr, 6 classes)")
    return model


def build_random_forest() -> RandomForestClassifier:
    model = RandomForestClassifier(**RF_PARAMS)
    print(f"  🌲 Random Forest created (n_estimators={RF_PARAMS['n_estimators']}, 6 classes)")
    return model


def build_gradient_boosting() -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(**GB_PARAMS)
    print(f"  🚀 Gradient Boosting created (n_estimators={GB_PARAMS['n_estimators']}, 6 classes)")
    return model
