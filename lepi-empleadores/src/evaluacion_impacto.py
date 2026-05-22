"""
evaluacion_impacto.py
Análisis estadístico de impacto LEPI: pruebas t, tamaño de efecto (Cohen's d),
intervalos de confianza y visualizaciones antes/después.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula el tamaño de efecto de Cohen's d."""
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0


def intervalo_confianza(datos: np.ndarray, alpha: float = 0.05):
    """IC al 95% para la media."""
    n   = len(datos)
    m   = datos.mean()
    sem = stats.sem(datos)
    h   = sem * stats.t.ppf(1 - alpha/2, n-1)
    return m - h, m + h


def analisis_antes_despues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prueba t pareada + Cohen's d + IC95% para cada indicador LEPI.
    Muestra tabla profesional de significancia estadística.
    """
    indicadores = {
        "ILD (Liderazgo)":    ("antes_ILD",  "despues_ILD"),
        "IRPD (Praxeología)": ("antes_IRPD", "despues_IRPD"),
        "IIP (Innovación)":   ("antes_IIP",  "despues_IIP"),
        "ILEPI (Global)":     ("ILEPI_antes","ILEPI_despues"),
        "Satisfacción":       ("antes_satisfaccion","despues_satisfaccion"),
    }

    filas = []
    print("\n" + "="*75)
    print("  ANÁLISIS ESTADÍSTICO DE IMPACTO — APP LEPI")
    print("="*75)
    print(f"{'Indicador':<22} {'Antes':>7} {'Después':>9} {'Δ':>7} "
          f"{'d Cohen':>9} {'p-valor':>10} {'Sig.'}")
    print("─"*75)

    for nombre, (ca, cd) in indicadores.items():
        if ca not in df.columns or cd not in df.columns:
            continue
        a    = df[ca].values
        d    = df[cd].values
        t, p = stats.ttest_rel(d, a)
        dc   = cohens_d(a, d)
        delta = d.mean() - a.mean()
        sig  = ("✅***" if p < 0.001 else
                "✅**"  if p < 0.01  else
                "✅*"   if p < 0.05  else "⚠️ n.s.")
        ef   = ("Grande" if abs(dc) > 0.8 else
                "Medio"  if abs(dc) > 0.5 else "Pequeño")
        print(f"{nombre:<22} {a.mean():>7.3f} {d.mean():>9.3f} "
              f"{delta:>+7.3f} {dc:>8.3f} ({ef}) {p:>8.4f} {sig}")
        filas.append({"indicador": nombre, "antes": a.mean(),
                      "despues": d.mean(), "delta": delta,
                      "cohens_d": dc, "p_valor": p, "sig": sig})

    print("─"*75)
    print("Significancia: *** p<0.001 | ** p<0.01 | * p<0.05 | n.s. no significativo")
    return pd.DataFrame(filas)


def graficar_impacto(df: pd.DataFrame, df_stats: pd.DataFrame):
    """Visualización profesional del impacto antes/después."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Evaluación de Impacto LEPI — Análisis Estadístico",
                 fontsize=14, fontweight="bold")

    colores = ["#E74C3C", "#27AE60"]
    items = df_stats[df_stats["indicador"] != "Satisfacción"].head(4)

    for i, (_, row) in enumerate(items.iterrows()):
        ax = axes.flat[i]
        bars = ax.bar(["Antes","Después"], [row["antes"],row["despues"]],
                      color=colores, edgecolor="white", width=0.5, alpha=0.85)
        ax.set_ylim(0, 5)
        for bar, val in zip(bars, [row["antes"],row["despues"]]):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                    f"{val:.3f}", ha="center", fontweight="bold")
        sig_label = row["sig"]
        ax.set_title(f"{row['indicador']}\n"
                     f"Δ={row['delta']:+.3f}  d={row['cohens_d']:.2f}  {sig_label}",
                     fontsize=10)
        ax.set_ylabel("Puntuación (1-5)")

    # Panel 5 — Δ por institución
    ax5 = axes[1, 1]
    cmap  = plt.cm.RdYlGn
    norma = plt.Normalize(df["delta_ILEPI"].min(), df["delta_ILEPI"].max())
    bars5 = ax5.bar(df["institucion"], df["delta_ILEPI"],
                    color=cmap(norma(df["delta_ILEPI"])))
    ax5.axhline(df["delta_ILEPI"].mean(), color="blue", linestyle="--",
                label=f"Media Δ={df['delta_ILEPI'].mean():.3f}")
    ax5.set_title("Δ ILEPI por Institución")
    ax5.set_ylabel("Δ ILEPI"); ax5.legend()
    ax5.tick_params(axis="x", rotation=45)

    # Panel 6 — Cohen's d
    ax6 = axes[1, 2]
    d_vals = df_stats["cohens_d"]
    ax6.barh(df_stats["indicador"], d_vals,
             color=["#27AE60" if v >= 0.8 else
                    "#F39C12" if v >= 0.5 else "#E74C3C" for v in d_vals])
    ax6.axvline(0.8, color="green", linestyle="--", alpha=0.7, label="Efecto grande (>0.8)")
    ax6.axvline(0.5, color="orange", linestyle="--", alpha=0.7, label="Efecto medio (>0.5)")
    ax6.set_title("Tamaño de Efecto (Cohen's d)")
    ax6.set_xlabel("d de Cohen"); ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def tabla_roi_educativo(df: pd.DataFrame):
    """Calcula el ROI educativo del programa LEPI."""
    delta_mean = df["delta_ILEPI"].mean()
    n_doc      = df["docentes"].sum()
    pct_mejora = delta_mean / 5.0 * 100

    print("\n" + "="*50)
    print("  ROI EDUCATIVO — PROGRAMA LEPI")
    print("="*50)
    print(f"  Δ ILEPI promedio     : +{delta_mean:.3f} puntos")
    print(f"  % de mejora          : +{pct_mejora:.1f}%")
    print(f"  Docentes impactados  : {n_doc:,}")
    print(f"  Instituciones        : {len(df)}")
    print(f"  Mayor impacto en     : {df.loc[df['delta_ILEPI'].idxmax(),'institucion']}")
    print("="*50)
