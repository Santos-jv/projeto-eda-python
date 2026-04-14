"""
enhanced_modeling.py
--------------------
Versão melhorada do pipeline de modelagem com Feature Engineering 
e Otimização de Hiperparâmetros (Grid Search).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importamos o pipeline original de processamento
from data_processing import pipeline

def engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas variáveis para ajudar o modelo a entender melhor os dados.
    """
    df_eng = df.copy()
    
    # 1. Renda por Dependente (evita divisão por zero)
    df_eng['renda_por_dependente'] = df_eng['renda_mensal'] / (df_eng['n_dependentes'] + 1)
    
    # 2. Interação entre Idade e Horas de Trabalho
    # Pode indicar senioridade ou esforço produtivo
    df_eng['esforço_produtivo'] = df_eng['idade'] * df_eng['horas_trabalho_semana']
    
    # 3. Mapeamento de Escolaridade (Ordinal)
    # Transforma categorias em pesos numéricos lógicos
    ordem_edu = {"Fundamental": 1, "Médio": 2, "Superior": 3, "Pós-graduação": 4}
    df_eng['edu_nivel'] = df_eng['escolaridade'].map(ordem_edu)
    
    return df_eng

def preparar_dados_avancado(df: pd.DataFrame):
    """Prepara features e encoders, incluindo as novas colunas."""
    df_model = engineering_features(df)
    
    encoders = {}
    # Codificamos apenas as colunas que ainda são texto
    for col in ["regiao"]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    # Alvo (Categoria de Renda)
    le_target = LabelEncoder()
    df_model["categoria_renda"] = le_target.fit_transform(df_model["categoria_renda"])
    
    # Definimos quais colunas usar no treino
    features = [
        "idade", "horas_trabalho_semana", "n_dependentes", "regiao", 
        "renda_por_dependente", "esforço_produtivo", "edu_nivel"
    ]
    
    X = df_model[features]
    y = df_model["categoria_renda"]
    
    return X, y, features

def otimizar_random_forest(X_train, y_train):
    """Executa Grid Search para encontrar os melhores parâmetros."""
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    print(">>> Iniciando Grid Search (isso pode demorar alguns segundos)...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores Parâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def executar_pipeline_melhorado():
    # 1. Obter dados
    df = pipeline(salvar_csv=False)
    
    # 2. Feature Engineering e Separação
    X, y, feature_names = preparar_dados_avancado(df)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Treino com Otimização
    modelo_otimizado = otimizar_random_forest(X_train, y_train)
    
    # 5. Avaliação
    y_pred = modelo_otimizado.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"NOVA ACURÁCIA: {acc:.2%}")
    print("="*30)
    print(classification_report(y_test, y_pred))

    # 6. Importância das Features
    importancias = pd.Series(modelo_otimizado.feature_importances_, index=feature_names).sort_values()
    importancias.plot(kind='barh', color='teal')
    plt.title("Importância das Features (Modelo Otimizado)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    executar_pipeline_melhorado()