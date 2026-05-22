"""
data_generator.py + feature_engineering.py
Datos sintéticos realistas para indicadores de liderazgo docente LEPI.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# GENERADOR DE DATOS
# ─────────────────────────────────────────────────────────
def generar_docentes(n: int = 300) -> pd.DataFrame:
    areas   = ["Matemáticas","Lenguaje","Ciencias","Sociales","Inglés","TIC","Artes"]
    niveles = ["Primaria","Secundaria","Media"]
    insts   = [f"IE-{str(i).zfill(3)}" for i in range(1, 16)]
    form    = ["Pregrado","Especialización","Maestría","Doctorado"]

    df = pd.DataFrame({
        "docente_id":   [f"DOC-{str(i).zfill(4)}" for i in range(1, n+1)],
        "institucion":  np.random.choice(insts, n),
        "area":         np.random.choice(areas, n),
        "nivel":        np.random.choice(niveles, n, p=[0.35, 0.45, 0.20]),
        "formacion":    np.random.choice(form, n, p=[0.25, 0.35, 0.30, 0.10]),
        "anos_exp":     np.random.randint(1, 30, n),
        "edad":         np.random.randint(25, 60, n),
        # Liderazgo (escala 1-5)
        "lid_vision":         np.round(np.random.normal(3.5, 0.8, n).clip(1, 5), 1),
        "lid_comunicacion":   np.round(np.random.normal(3.7, 0.7, n).clip(1, 5), 1),
        "lid_colaboracion":   np.round(np.random.normal(3.6, 0.9, n).clip(1, 5), 1),
        "lid_motivacion":     np.round(np.random.normal(3.8, 0.7, n).clip(1, 5), 1),
        "lid_resolucion":     np.round(np.random.normal(3.4, 0.8, n).clip(1, 5), 1),
        # Praxeología
        "prax_ver":      np.round(np.random.normal(3.6, 0.7, n).clip(1, 5), 1),
        "prax_juzgar":   np.round(np.random.normal(3.3, 0.8, n).clip(1, 5), 1),
        "prax_actuar":   np.round(np.random.normal(3.5, 0.7, n).clip(1, 5), 1),
        "prax_devolver": np.round(np.random.normal(3.2, 0.9, n).clip(1, 5), 1),
        # Innovación
        "inn_tecnologia":  np.round(np.random.normal(3.4, 0.9, n).clip(1, 5), 1),
        "inn_metodologia": np.round(np.random.normal(3.5, 0.8, n).clip(1, 5), 1),
        "inn_evaluacion":  np.round(np.random.normal(3.3, 0.7, n).clip(1, 5), 1),
        # Contexto
        "satisfaccion":    np.round(np.random.normal(3.8, 0.7, n).clip(1, 5), 1),
        "carga_laboral":   np.round(np.random.normal(3.5, 0.8, n).clip(1, 5), 1),
        "horas_formacion": np.random.randint(0, 120, n),
        "evaluacion_par":  np.round(np.random.normal(3.6, 0.7, n).clip(1, 5), 1),
    })
    return agregar_indices(df)


def generar_impacto(n: int = 15) -> pd.DataFrame:
    """Dataset antes/después para evaluación de impacto LEPI."""
    insts = [f"IE-{str(i).zfill(3)}" for i in range(1, n+1)]
    df = pd.DataFrame({
        "institucion":         insts,
        "zona":                np.random.choice(["Urbana","Rural"], n, p=[0.7, 0.3]),
        "docentes":            np.random.randint(15, 80, n),
        "antes_ILD":           np.round(np.random.normal(2.9, 0.3, n).clip(2.0, 3.8), 2),
        "antes_IRPD":          np.round(np.random.normal(2.8, 0.3, n).clip(2.0, 3.6), 2),
        "antes_IIP":           np.round(np.random.normal(2.7, 0.3, n).clip(1.8, 3.5), 2),
        "antes_satisfaccion":  np.round(np.random.normal(3.1, 0.3, n).clip(2.0, 4.0), 2),
        "despues_ILD":         np.round(np.random.normal(3.9, 0.2, n).clip(3.3, 4.8), 2),
        "despues_IRPD":        np.round(np.random.normal(3.8, 0.2, n).clip(3.2, 4.7), 2),
        "despues_IIP":         np.round(np.random.normal(3.7, 0.2, n).clip(3.0, 4.6), 2),
        "despues_satisfaccion":np.round(np.random.normal(4.1, 0.2, n).clip(3.4, 4.9), 2),
    })
    df["delta_ILD"]  = (df["despues_ILD"]  - df["antes_ILD"]).round(3)
    df["delta_IRPD"] = (df["despues_IRPD"] - df["antes_IRPD"]).round(3)
    df["delta_IIP"]  = (df["despues_IIP"]  - df["antes_IIP"]).round(3)
    df["ILEPI_antes"]   = (df["antes_ILD"]*0.35 + df["antes_IRPD"]*0.35 + df["antes_IIP"]*0.30).round(3)
    df["ILEPI_despues"] = (df["despues_ILD"]*0.35 + df["despues_IRPD"]*0.35 + df["despues_IIP"]*0.30).round(3)
    df["delta_ILEPI"]   = (df["ILEPI_despues"] - df["ILEPI_antes"]).round(3)
    return df


# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING — Índices compuestos LEPI
# ─────────────────────────────────────────────────────────
DIMS_ILD  = ["lid_vision","lid_comunicacion","lid_colaboracion","lid_motivacion","lid_resolucion"]
DIMS_IRPD = ["prax_ver","prax_juzgar","prax_actuar","prax_devolver"]
DIMS_IIP  = ["inn_tecnologia","inn_metodologia","inn_evaluacion"]
PESOS     = {"ILD": 0.35, "IRPD": 0.35, "IIP": 0.30}


def agregar_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ILD, IRPD, IIP e ILEPI. Feature engineering principal."""
    df = df.copy()
    # Índices simples
    df["ILD"]   = df[DIMS_ILD].mean(axis=1).round(3)
    df["IRPD"]  = df[DIMS_IRPD].mean(axis=1).round(3)
    df["IIP"]   = df[DIMS_IIP].mean(axis=1).round(3)
    # Índice compuesto ponderado
    df["ILEPI"] = (df["ILD"]*PESOS["ILD"] +
                   df["IRPD"]*PESOS["IRPD"] +
                   df["IIP"]*PESOS["IIP"]).round(3)
    # Variables derivadas adicionales
    df["ratio_form_exp"]  = (df["horas_formacion"] / (df["anos_exp"] + 1)).round(2)
    df["brecha_lid_inn"]  = (df["ILD"] - df["IIP"]).round(3)
    df["indice_bienestar"]= (df["satisfaccion"] * 0.6 + (5 - df["carga_laboral"]) * 0.4).round(3)
    # Etiquetas
    df["nivel_lepi"] = pd.cut(df["ILEPI"],
        bins=[0, 2.9, 3.4, 3.9, 5.0],
        labels=["⚠️ Inicial","📈 En Desarrollo","✅ Consolidado","🌟 Avanzado"])
    df["perfil_liderazgo"] = df["ILD"].apply(
        lambda x: "Transformacional" if x >= 4.0 else
                  ("Transaccional"   if x >= 3.0 else "Laissez-faire"))
    return df


def resumen_institucional(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("institucion").agg(
        docentes    = ("docente_id","count"),
        ILD_prom    = ("ILD","mean"),
        IRPD_prom   = ("IRPD","mean"),
        IIP_prom    = ("IIP","mean"),
        ILEPI_prom  = ("ILEPI","mean"),
        pct_avanzado= ("nivel_lepi", lambda x: (x=="🌟 Avanzado").mean()*100),
    ).round(3).reset_index().sort_values("ILEPI_prom", ascending=False)
