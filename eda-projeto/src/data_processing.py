"""
data_processing.py
------------------
Coleta, limpeza e tratamento do dataset socioeconômico.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Geração / Carregamento do dataset
# ---------------------------------------------------------------------------

def gerar_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Gera um dataset sintético com variáveis socioeconômicas."""
    rng = np.random.default_rng(seed)

    idades = rng.integers(18, 70, size=n).astype(float)
    rendas = rng.normal(loc=3500, scale=1500, size=n)
    escolaridade = rng.choice(
        ["Fundamental", "Médio", "Superior", "Pós-graduação"],
        size=n,
        p=[0.15, 0.40, 0.35, 0.10],
    )
    regiao = rng.choice(
        ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
        size=n,
        p=[0.08, 0.28, 0.09, 0.42, 0.13],
    )
    horas_trabalho = rng.integers(20, 60, size=n).astype(float)
    n_dependentes = rng.integers(0, 6, size=n)

    # Variável alvo: categoria de renda
    limites = [0, 2000, 4000, np.inf]
    rotulos = ["Baixa", "Média", "Alta"]
    categoria_renda = pd.cut(rendas, bins=limites, labels=rotulos)

    df = pd.DataFrame(
        {
            "idade": idades,
            "renda_mensal": rendas,
            "escolaridade": escolaridade,
            "regiao": regiao,
            "horas_trabalho_semana": horas_trabalho,
            "n_dependentes": n_dependentes,
            "categoria_renda": categoria_renda,
        }
    )

    # Introduz valores nulos (~12 %) e duplicatas artificiais
    null_idx_idade = rng.choice(df.index, size=int(n * 0.05), replace=False)
    null_idx_renda = rng.choice(df.index, size=int(n * 0.07), replace=False)
    df.loc[null_idx_idade, "idade"] = np.nan
    df.loc[null_idx_renda, "renda_mensal"] = np.nan

    duplicatas = df.sample(n=30, random_state=seed)
    df = pd.concat([df, duplicatas], ignore_index=True)

    return df


def carregar_dataset(caminho: str) -> pd.DataFrame:
    """Carrega dataset a partir de arquivo CSV ou JSON."""
    if caminho.endswith(".json"):
        return pd.read_json(caminho)
    return pd.read_csv(caminho)


# ---------------------------------------------------------------------------
# 2. Diagnóstico inicial
# ---------------------------------------------------------------------------

def diagnostico(df: pd.DataFrame) -> None:
    """Imprime um resumo diagnóstico do DataFrame."""
    print("=" * 55)
    print(f"Shape          : {df.shape}")
    print(f"Duplicatas     : {df.duplicated().sum()}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
    print(f"\nTipos:\n{df.dtypes}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# 3. Limpeza e tratamento
# ---------------------------------------------------------------------------

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    antes = len(df)
    df = df.drop_duplicates()
    print(f"[limpeza] Duplicatas removidas: {antes - len(df)}")
    return df


def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa mediana em numéricas e moda em categóricas."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"[nulos] '{col}' → imputado com mediana={mediana:.2f}")

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            print(f"[nulos] '{col}' → imputado com moda='{moda}'")

    return df


def remover_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Remove outliers pela regra do IQR (1.5×)."""
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    antes = len(df)
    df = df[(df[coluna] >= Q1 - 1.5 * IQR) & (df[coluna] <= Q3 + 1.5 * IQR)]
    print(f"[outliers] '{coluna}' → removidos {antes - len(df)} registros")
    return df


# ---------------------------------------------------------------------------
# 4. Estatística descritiva
# ---------------------------------------------------------------------------

def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com estatísticas descritivas das colunas numéricas."""
    numericas = df.select_dtypes(include=[np.number])
    stats = numericas.agg(["mean", "median", "std", "min", "max"]).T
    stats.columns = ["Média", "Mediana", "Desvio Padrão", "Mínimo", "Máximo"]
    return stats.round(2)


def matriz_correlacao(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a matriz de correlação das variáveis numéricas."""
    return df.select_dtypes(include=[np.number]).corr().round(3)


# ---------------------------------------------------------------------------
# 5. Pipeline completo
# ---------------------------------------------------------------------------

def pipeline(salvar_csv: bool = True) -> pd.DataFrame:
    print("\n>>> Gerando dataset...")
    df = gerar_dataset()

    print("\n>>> Diagnóstico inicial:")
    diagnostico(df)

    print("\n>>> Limpeza:")
    df = remover_duplicatas(df)
    df = tratar_nulos(df)
    df = remover_outliers_iqr(df, "renda_mensal")

    print("\n>>> Estatísticas descritivas:")
    print(estatisticas_descritivas(df).to_string())

    print("\n>>> Correlação:")
    print(matriz_correlacao(df).to_string())

    if salvar_csv:
        df.to_csv("data/dataset.csv", index=False)
        print("\n[✓] Dataset salvo em data/dataset.csv")

    return df


if __name__ == "__main__":
    pipeline()
