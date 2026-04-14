"""
visualization.py
----------------
Visualizações com Matplotlib e Seaborn para comunicar
padrões e insights a públicos não técnicos.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from data_processing import pipeline

# ---------------------------------------------------------------------------
# Configuração global de estilo
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 6)})
os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------------------
# Funções de visualização
# ---------------------------------------------------------------------------

def plot_distribuicao_renda(df: pd.DataFrame) -> None:
    """Histograma + KDE da renda mensal."""
    fig, ax = plt.subplots()
    sns.histplot(df["renda_mensal"], bins=40, kde=True, color="#4C72B0", ax=ax)
    ax.set_title("Distribuição da Renda Mensal", fontsize=14, fontweight="bold")
    ax.set_xlabel("Renda (R$)")
    ax.set_ylabel("Frequência")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    plt.tight_layout()
    plt.savefig("outputs/distribuicao_renda.png")
    plt.close()
    print("[✓] outputs/distribuicao_renda.png")


def plot_categoria_por_escolaridade(df: pd.DataFrame) -> None:
    """Gráfico de barras empilhadas: categoria de renda × escolaridade."""
    ordem_escol = ["Fundamental", "Médio", "Superior", "Pós-graduação"]
    ordem_cat = ["Baixa", "Média", "Alta"]
    tab = (
        df.groupby(["escolaridade", "categoria_renda"], observed=True)
        .size()
        .unstack("categoria_renda")[ordem_cat]
        .loc[ordem_escol]
    )
    ax = tab.plot(kind="bar", stacked=True, colormap="Blues", figsize=(10, 6))
    ax.set_title("Categoria de Renda por Escolaridade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Escolaridade")
    ax.set_ylabel("Número de pessoas")
    ax.legend(title="Categoria de Renda", bbox_to_anchor=(1.01, 1))
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/categoria_por_escolaridade.png")
    plt.close()
    print("[✓] outputs/categoria_por_escolaridade.png")


def plot_boxplot_renda_regiao(df: pd.DataFrame) -> None:
    """Boxplot da renda por região."""
    fig, ax = plt.subplots()
    ordem = df.groupby("regiao")["renda_mensal"].median().sort_values().index
    sns.boxplot(
        data=df,
        x="regiao",
        y="renda_mensal",
        order=ordem,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Renda Mensal por Região", fontsize=14, fontweight="bold")
    ax.set_xlabel("Região")
    ax.set_ylabel("Renda (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    plt.tight_layout()
    plt.savefig("outputs/boxplot_renda_regiao.png")
    plt.close()
    print("[✓] outputs/boxplot_renda_regiao.png")


def plot_heatmap_correlacao(df: pd.DataFrame) -> None:
    """Heatmap da matriz de correlação."""
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/heatmap_correlacao.png")
    plt.close()
    print("[✓] outputs/heatmap_correlacao.png")


def plot_scatterplot_idade_renda(df: pd.DataFrame) -> None:
    """Scatter plot: idade × renda, colorido por categoria."""
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="idade",
        y="renda_mensal",
        hue="categoria_renda",
        hue_order=["Baixa", "Média", "Alta"],
        palette="deep",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Idade × Renda por Categoria", fontsize=14, fontweight="bold")
    ax.set_xlabel("Idade (anos)")
    ax.set_ylabel("Renda (R$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax.legend(title="Categoria")
    plt.tight_layout()
    plt.savefig("outputs/scatter_idade_renda.png")
    plt.close()
    print("[✓] outputs/scatter_idade_renda.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot das variáveis numéricas."""
    numericas = ["idade", "renda_mensal", "horas_trabalho_semana", "n_dependentes"]
    g = sns.pairplot(
        df[numericas + ["categoria_renda"]],
        hue="categoria_renda",
        hue_order=["Baixa", "Média", "Alta"],
        palette="deep",
        diag_kind="kde",
        plot_kws={"alpha": 0.5},
    )
    g.fig.suptitle("Pairplot – Variáveis Socioeconômicas", y=1.02, fontsize=14, fontweight="bold")
    plt.savefig("outputs/pairplot.png", bbox_inches="tight")
    plt.close()
    print("[✓] outputs/pairplot.png")


# ---------------------------------------------------------------------------
# Pipeline de visualizações
# ---------------------------------------------------------------------------

def gerar_todas_visualizacoes(df: pd.DataFrame) -> None:
    print("\n>>> Gerando visualizações...")
    plot_distribuicao_renda(df)
    plot_categoria_por_escolaridade(df)
    plot_boxplot_renda_regiao(df)
    plot_heatmap_correlacao(df)
    plot_scatterplot_idade_renda(df)
    plot_pairplot(df)
    print("\n[✓] Todas as visualizações salvas em outputs/")


if __name__ == "__main__":
    df = pipeline(salvar_csv=True)
    gerar_todas_visualizacoes(df)
