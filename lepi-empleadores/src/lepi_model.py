"""
lepi_model.py
Clase MLPipeline que encapsula todos los modelos LEPI.
Diseñado para demostrar buenas prácticas de ingeniería ML a empleadores.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestClassifier)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error,
                              roc_auc_score, roc_curve,
                              classification_report,
                              silhouette_score)
import warnings
warnings.filterwarnings("ignore")


FEATURES_REG = ["anos_exp","edad","lid_vision","lid_comunicacion",
                "prax_ver","prax_juzgar","inn_tecnologia",
                "satisfaccion","carga_laboral","ratio_form_exp",
                "horas_formacion","evaluacion_par"]

FEATURES_CLF = FEATURES_REG + ["ILD","IRPD","IIP"]


class LEPIMLPipeline:
    """
    Pipeline ML completo para el sistema LEPI.
    Incluye: clustering de perfiles, regresión ILEPI y clasificación de alertas.
    """

    def __init__(self):
        self.pipeline_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=3, subsample=0.8, random_state=42))
        ])
        self.pipeline_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators=200, max_depth=6,
                class_weight="balanced", random_state=42))
        ])
        self.kmeans  = None
        self.scaler_km = StandardScaler()
        self.trained = False
        self.metricas = {}

    # ── Clustering ───────────────────────────────────────
    def entrenar_clustering(self, df: pd.DataFrame,
                            k: int = 4) -> pd.DataFrame:
        """K-Means para segmentar perfiles docentes."""
        features = ["ILD","IRPD","IIP"]
        X = self.scaler_km.fit_transform(df[features])
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = self.kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        self.metricas["silhouette"] = round(sil, 4)
        print(f"  K-Means K={k} | Silhouette = {sil:.4f}  {'✅' if sil>0.5 else '⚠️'}")
        df = df.copy()
        df["cluster"] = labels
        # Etiquetar automáticamente por centroide
        centros = pd.DataFrame(
            self.scaler_km.inverse_transform(self.kmeans.cluster_centers_),
            columns=features)
        ilepi_c = centros.mean(axis=1)
        order   = ilepi_c.rank(ascending=False).astype(int)
        mapa    = {order[order==1].index[0]: "🏆 Líder LEPI",
                   order[order==2].index[0]: "🧠 Reflexivo",
                   order[order==3].index[0]: "💡 Innovador",
                   order[order==4].index[0]: "📚 En Formación"}
        df["perfil_cluster"] = df["cluster"].map(mapa)
        return df

    # ── Regresión: predicción ILEPI ──────────────────────
    def entrenar_regresion(self, df: pd.DataFrame) -> dict:
        """Predice el ILEPI futuro del docente."""
        X = df[FEATURES_REG]; y = df["ILEPI"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.pipeline_reg.fit(X_tr, y_tr)
        y_pred = self.pipeline_reg.predict(X_te)
        r2  = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        cv  = cross_val_score(self.pipeline_reg, X, y, cv=5, scoring="r2")
        self.metricas.update({"R2": round(r2,4), "MAE": round(mae,4),
                              "CV_R2_mean": round(cv.mean(),4),
                              "CV_R2_std":  round(cv.std(),4)})
        print(f"  Regresión ILEPI  | R²={r2:.4f}  MAE={mae:.4f}  "
              f"CV={cv.mean():.4f}±{cv.std():.4f}  {'✅' if r2>0.75 else '⚠️'}")
        self._y_te_reg = y_te; self._y_pred_reg = y_pred
        self._X_te_reg = X_te
        return self.metricas

    # ── Clasificación: alerta desempeño bajo ─────────────
    def entrenar_clasificacion(self, df: pd.DataFrame) -> dict:
        """Clasifica docentes con riesgo de bajo desempeño."""
        y_target = (df["ILEPI"] < 3.0).astype(int)
        X = df[FEATURES_CLF]
        X_tr,X_te,y_tr,y_te = train_test_split(
            X, y_target, test_size=0.2, random_state=42, stratify=y_target)
        self.pipeline_clf.fit(X_tr, y_tr)
        y_prob = self.pipeline_clf.predict_proba(X_te)[:,1]
        y_pred = self.pipeline_clf.predict(X_te)
        auc = roc_auc_score(y_te, y_prob)
        self.metricas["AUC"] = round(auc, 4)
        print(f"  Clasificación    | AUC={auc:.4f}  {'✅' if auc>0.8 else '⚠️'}")
        print(classification_report(y_te, y_pred,
              target_names=["Desempeño OK","Riesgo bajo"]))
        self._y_te_clf = y_te; self._y_prob_clf = y_prob; self._y_pred_clf = y_pred
        return self.metricas

    # ── Importancia de variables ─────────────────────────
    def importancia_variables(self) -> pd.Series:
        """Extrae importancia de variables del modelo de regresión."""
        model = self.pipeline_reg.named_steps["model"]
        imp = pd.Series(model.feature_importances_,
                        index=FEATURES_REG).sort_values(ascending=False)
        return imp

    # ── Predicción individual ────────────────────────────
    def predecir_docente(self, datos: dict) -> dict:
        """Predice ILEPI y riesgo de un docente nuevo."""
        df_d = pd.DataFrame([datos])
        ilepi_pred = self.pipeline_reg.predict(df_d[FEATURES_REG])[0]
        prob_riesgo = self.pipeline_clf.predict_proba(df_d[FEATURES_CLF])[0][1]
        alerta = ("🔴 Riesgo Alto"  if prob_riesgo > 0.7 else
                  ("🟡 Riesgo Medio" if prob_riesgo > 0.4 else "🟢 Óptimo"))
        return {"ILEPI_predicho": round(ilepi_pred, 3),
                "prob_riesgo":    round(prob_riesgo, 4),
                "alerta":         alerta}

    # ── Dashboard de métricas ────────────────────────────
    def dashboard_metricas(self):
        """Panel visual de métricas de todos los modelos."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Dashboard ML — App LEPI | Portafolio Analista de Datos",
                     fontsize=14, fontweight="bold")

        # 1. Predicción vs Real
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(self._y_te_reg, self._y_pred_reg,
                    alpha=0.6, color="#2E75B6", s=30)
        lims = [min(self._y_te_reg.min(), self._y_pred_reg.min()),
                max(self._y_te_reg.max(), self._y_pred_reg.max())]
        ax1.plot(lims, lims, "r--", lw=2)
        ax1.set_xlabel("ILEPI Real"); ax1.set_ylabel("ILEPI Predicho")
        ax1.set_title(f"Predicción ILEPI\nR²={self.metricas['R2']:.3f}  MAE={self.metricas['MAE']:.3f}")

        # 2. Curva ROC
        ax2 = fig.add_subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self._y_te_clf, self._y_prob_clf)
        ax2.plot(fpr, tpr, color="green", lw=2,
                 label=f"AUC={self.metricas['AUC']:.3f}")
        ax2.plot([0,1],[0,1],"r--"); ax2.fill_between(fpr,tpr,alpha=0.1,color="green")
        ax2.set_title("Curva ROC — Alerta Docente")
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()

        # 3. Importancia de variables
        ax3 = fig.add_subplot(2, 3, 3)
        imp = self.importancia_variables().head(8)
        ax3.barh(imp.index, imp.values,
                 color=plt.cm.Blues(np.linspace(0.4, 1.0, 8)))
        ax3.set_title("Top 8 Variables Más Importantes")

        # 4. Residuos
        ax4 = fig.add_subplot(2, 3, 4)
        res = self._y_te_reg.values - self._y_pred_reg
        ax4.hist(res, bins=25, color="steelblue", edgecolor="white")
        ax4.axvline(0, color="red", linestyle="--")
        ax4.set_title(f"Residuos del Modelo\nMedia={res.mean():.4f}")

        # 5. Métricas resumen
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis("off")
        tabla = [
            ["Modelo", "Técnica", "Métrica", "Valor"],
            ["Perfiles","K-Means K=4","Silhouette",str(self.metricas.get("silhouette","—"))],
            ["ILEPI","Gradient Boosting","R²",str(self.metricas.get("R2","—"))],
            ["ILEPI","Cross-Val (5k)","R² CV",f"{self.metricas.get('CV_R2_mean','—')}±{self.metricas.get('CV_R2_std','—')}"],
            ["Alerta","Random Forest","AUC",str(self.metricas.get("AUC","—"))],
        ]
        t = ax5.table(cellText=tabla[1:], colLabels=tabla[0],
                      loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(10)
        t.scale(1.2, 1.8)
        for j in range(4):
            t[0,j].set_facecolor("#1F3864")
            t[0,j].set_text_props(color="white", fontweight="bold")
        ax5.set_title("Resumen de Métricas ML", fontweight="bold", pad=80)

        # 6. CV scores
        ax6 = fig.add_subplot(2, 3, 6)
        cv_scores = cross_val_score(self.pipeline_reg,
                                    self._X_te_reg,
                                    self._y_te_reg, cv=5, scoring="r2")
        ax6.bar(range(1,6), cv_scores, color="#2E75B6", alpha=0.85)
        ax6.axhline(cv_scores.mean(), color="red", linestyle="--",
                    label=f"Media={cv_scores.mean():.3f}")
        ax6.set_title("Cross-Validation R² (5 folds)")
        ax6.set_xlabel("Fold"); ax6.set_ylabel("R²"); ax6.legend()

        plt.tight_layout()
        plt.show()
