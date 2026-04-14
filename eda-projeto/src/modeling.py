"""
modeling.py
-----------
Treinamento e avaliação do modelo Random Forest para
classificação de categoria de renda.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_processing import pipeline

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------------------
# Pré-processamento para modelagem
# ---------------------------------------------------------------------------

def preparar_features(df: pd.DataFrame):
    """
    Codifica variáveis categóricas e separa features (X) do alvo (y).
    Retorna X, y e os encoders utilizados.
    """
    df_model = df.copy()

    # Codificação de variáveis categóricas nominais
    encoders = {}
    for col in ["escolaridade", "regiao"]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    # Alvo
    le_target = LabelEncoder()
    df_model["categoria_renda"] = le_target.fit_transform(
        df_model["categoria_renda"].astype(str)
    )
    encoders["categoria_renda"] = le_target

    feature_cols = [
        "idade",
        "renda_mensal",
        "escolaridade",
        "regiao",
        "horas_trabalho_semana",
        "n_dependentes",
    ]

    X = df_model[feature_cols]
    y = df_model["categoria_renda"]
    return X, y, encoders, feature_cols


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def treinar_modelo(X_train, y_train, seed: int = 42) -> RandomForestClassifier:
    """Treina e retorna um RandomForestClassifier."""
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def avaliar_modelo(
    modelo: RandomForestClassifier,
    X_train,
    X_test,
    y_train,
    y_test,
    encoders: dict,
    feature_cols: list,
) -> None:
    """Imprime métricas e salva gráficos de avaliação."""

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    # Validação cruzada (5-fold)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="accuracy")

    classes = encoders["categoria_renda"].classes_

    print("\n" + "=" * 55)
    print(f"  Acurácia (test set)   : {acuracia:.4f}  ({acuracia*100:.1f}%)")
    print(f"  CV 5-fold – média     : {cv_scores.mean():.4f}")
    print(f"  CV 5-fold – std       : {cv_scores.std():.4f}")
    print("=" * 55)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Matriz de Confusão – Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("[✓] outputs/confusion_matrix.png")

    # --- Feature importance ---
    importancias = pd.Series(modelo.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importancias.values, y=importancias.index, palette="Blues_r", ax=ax)
    ax.set_title("Importância das Features – Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância Média (Gini)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()
    print("[✓] outputs/feature_importance.png")


# ---------------------------------------------------------------------------
# Pipeline completo de modelagem
# ---------------------------------------------------------------------------

def pipeline_modelagem() -> None:
    # 1. Dados
    df = pipeline(salvar_csv=True)

    # 2. Features
    X, y, encoders, feature_cols = preparar_features(df)

    # 3. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[dados] Treino: {len(X_train)} | Teste: {len(X_test)}")

    # 4. Treino
    print("\n>>> Treinando Random Forest...")
    modelo = treinar_modelo(X_train, y_train)

    # 5. Avaliação
    avaliar_modelo(modelo, X_train, X_test, y_train, y_test, encoders, feature_cols)


if __name__ == "__main__":
    pipeline_modelagem()
