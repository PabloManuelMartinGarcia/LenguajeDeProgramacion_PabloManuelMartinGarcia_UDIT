"""Simulacion de produccion del modelo de superventas.

Carga una sola vez el modelo entrenado (``modelo_superventas.pkl``), como haria
un servicio al arrancar, y despues atiende "peticiones": valida los datos de
entrada, predice y registra cada respuesta en un log.

Uso:

    python produccion.py                       # simula una tanda de peticiones de ejemplo
    python produccion.py --interactivo         # escribes tu los datos del juego
    python produccion.py --entrada juegos.csv  # prediccion por lotes de un CSV
    python produccion.py --evaluar             # genera el informe HTML con todas las graficas

El CSV necesita las columnas Year, Platform, Genre y Publisher (Name es opcional).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_superventas.pkl"
RUTA_LOG = CARPETA / "logs" / "produccion.log"

# Peticiones de ejemplo que "llegan" al servicio (la ultima es incorrecta a proposito)
PETICIONES_EJEMPLO: list[dict[str, Any]] = [
    {"Name": "Shooter de Activision", "Year": 2016, "Platform": "PS4", "Genre": "Shooter", "Publisher": "Activision"},
    {"Name": "Plataformas de Nintendo", "Year": 2016, "Platform": "WiiU", "Genre": "Platform", "Publisher": "Nintendo"},
    {"Name": "Deportes de EA", "Year": 2016, "Platform": "XOne", "Genre": "Sports", "Publisher": "Electronic Arts"},
    {"Name": "Aventura indie", "Year": 2016, "Platform": "PC", "Genre": "Adventure", "Publisher": "Estudio Indie"},
    {"Name": "Peticion con error", "Year": 1850, "Platform": "PS4", "Genre": "", "Publisher": "Sony"},
]


def configurar_log() -> logging.Logger:
    """Escribe los mensajes en la consola y en ``logs/produccion.log``."""
    RUTA_LOG.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("produccion")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formato = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(RUTA_LOG, encoding="utf-8")):
            handler.setFormatter(formato)
            logger.addHandler(handler)
    return logger


class ErrorValidacion(ValueError):
    """Los datos de la peticion no son validos."""


class ServicioSuperventas:
    """Servicio que carga el modelo una vez y responde a peticiones de prediccion."""

    def __init__(self, ruta_modelo: Path, logger: logging.Logger) -> None:
        """Carga el ``.pkl`` (equivale al arranque del servicio).

        Args:
            ruta_modelo: Ruta del archivo ``.pkl`` exportado desde el notebook.
            logger: Logger donde se registran las peticiones.

        Raises:
            FileNotFoundError: Si no existe el ``.pkl``.
        """
        if not ruta_modelo.is_file():
            raise FileNotFoundError(
                f"No se encontro {ruta_modelo.name}. Ejecuta antes el notebook eda_y_entrenamiento.ipynb."
            )
        inicio = time.perf_counter()
        with open(ruta_modelo, "rb") as archivo:
            artefacto = pickle.load(archivo)  # solo cargar .pkl propios: pickle puede ejecutar codigo

        if isinstance(artefacto, dict):
            self.modelo = artefacto["modelo"]
            self.variables: list[str] = artefacto["variables"]
            self.umbral: float = artefacto["umbral"]
            version_entrenamiento = artefacto.get("sklearn_version")
            fecha_entrenamiento = artefacto.get("fecha_entrenamiento", "desconocida")
            metricas = artefacto.get("metricas", {})
        else:
            self.modelo = artefacto
            self.variables = ["Year", "Platform", "Genre", "Publisher"]
            self.umbral = 0.5
            version_entrenamiento = None
            fecha_entrenamiento = "desconocida"
            metricas = {}
        self.logger = logger

        # Categorias que el modelo vio al entrenar, para avisar si llega una desconocida
        nombre_preprocesador = next(
            (nombre for nombre in ("preprocesador", "preprocessor") if nombre in self.modelo.named_steps),
            None,
        )
        if nombre_preprocesador is None:
            raise ValueError("El modelo del .pkl no contiene un paso de preprocesamiento reconocido.")
        encoder = self.modelo.named_steps[nombre_preprocesador].named_transformers_["cat"].named_steps["encoder"]
        self.categorias_conocidas = {
            columna: set(valores) for columna, valores in zip(["Platform", "Genre", "Publisher"], encoder.categories_)
        }

        if version_entrenamiento and version_entrenamiento != sklearn.__version__:
            logger.warning(
                "Modelo entrenado con scikit-learn %s y ejecutado con %s: podria haber diferencias.",
                version_entrenamiento, sklearn.__version__,
            )
        logger.info(
            "Servicio arrancado en %.0f ms · modelo del %s · metricas en test: %s",
            (time.perf_counter() - inicio) * 1000, fecha_entrenamiento, metricas,
        )

    def validar(self, peticion: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        """Comprueba los datos de entrada y los convierte al formato del modelo.

        Args:
            peticion: Diccionario con Year, Platform, Genre y Publisher.

        Returns:
            Una fila lista para el modelo y la lista de avisos (categorias desconocidas).

        Raises:
            ErrorValidacion: Si falta un campo o algun valor no es valido.
        """
        faltan = [v for v in self.variables if v not in peticion]
        if faltan:
            raise ErrorValidacion(f"faltan los campos {faltan}")
        try:
            ano = int(peticion["Year"])
        except (TypeError, ValueError):
            raise ErrorValidacion(f"Year debe ser un numero entero (llego {peticion['Year']!r})") from None
        if not 1970 <= ano <= 2035:
            raise ErrorValidacion(f"Year={ano} fuera del rango permitido (1970-2035)")

        fila = {"Year": ano}
        avisos = []
        for campo in ["Platform", "Genre", "Publisher"]:
            valor = str(peticion[campo]).strip()
            if not valor:
                raise ErrorValidacion(f"{campo} no puede estar vacio")
            if valor not in self.categorias_conocidas[campo]:
                avisos.append(f"{campo}='{valor}' no aparece en los datos de entrenamiento")
            fila[campo] = valor
        if fila["Genre"] not in self.categorias_conocidas["Genre"]:
            raise ErrorValidacion(f"Genre='{fila['Genre']}' no valido. Opciones: {sorted(self.categorias_conocidas['Genre'])}")
        return pd.DataFrame([fila], columns=self.variables), avisos

    def predecir(self, peticion: dict[str, Any]) -> dict[str, Any]:
        """Atiende una peticion: valida, predice y registra el resultado.

        Args:
            peticion: Datos del juego.

        Returns:
            Respuesta con la probabilidad y la prediccion, o con el error si la peticion no era valida.
        """
        inicio = time.perf_counter()
        nombre = peticion.get("Name", "sin nombre")
        try:
            fila, avisos = self.validar(peticion)
        except ErrorValidacion as error:
            self.logger.error("'%s' rechazada: %s", nombre, error)
            return {"Name": nombre, "estado": "error", "detalle": str(error)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # aviso de sklearn por categorias desconocidas: ya se informa abajo
            probabilidad = float(self.modelo.predict_proba(fila)[0, 1])
        prediccion = "Superventas" if probabilidad >= self.umbral else "No superventas"
        ms = (time.perf_counter() - inicio) * 1000

        for aviso in avisos:
            self.logger.warning("'%s': %s (se trata como categoria poco frecuente)", nombre, aviso)
        self.logger.info("'%s' → %s (%.0f %%) en %.1f ms", nombre, prediccion, probabilidad * 100, ms)
        return {
            "Name": nombre, **fila.iloc[0].to_dict(), "estado": "ok",
            "probabilidad": round(probabilidad, 3), "prediccion": prediccion, "latencia_ms": round(ms, 1),
        }


def modo_ejemplo(servicio: ServicioSuperventas) -> list[dict[str, Any]]:
    """Simula que van llegando peticiones al servicio."""
    respuestas = []
    for peticion in PETICIONES_EJEMPLO:
        respuestas.append(servicio.predecir(peticion))
        time.sleep(0.3)  # tiempo entre peticiones, para que se vea el flujo
    mostrar_resumen(respuestas)
    return respuestas


def modo_lotes(servicio: ServicioSuperventas, ruta_csv: Path) -> list[dict[str, Any]]:
    """Predice todos los juegos de un CSV y guarda el resultado."""
    peticiones = pd.read_csv(ruta_csv).to_dict(orient="records")
    respuestas = [servicio.predecir(p) for p in peticiones]
    salida = ruta_csv.with_name(ruta_csv.stem + "_predicciones.csv")
    pd.DataFrame(respuestas).to_csv(salida, index=False)
    mostrar_resumen(respuestas)
    print(f"Resultados guardados en: {salida}")
    return respuestas


def modo_interactivo(servicio: ServicioSuperventas) -> list[dict[str, Any]]:
    """Pide los datos de un juego por teclado hasta que se pulsa Enter sin escribir nada."""
    print("\nEscribe los datos del juego (Enter vacío en 'Nombre' para salir).")
    print("Géneros válidos:", ", ".join(sorted(servicio.categorias_conocidas["Genre"])))
    respuestas = []
    while True:
        nombre = input("\nNombre: ").strip()
        if not nombre:
            break
        peticion = {
            "Name": nombre,
            "Year": input("Año (ej. 2016): ").strip(),
            "Platform": input("Plataforma (ej. PS4, XOne, PC, 3DS): ").strip(),
            "Genre": input("Género (ej. Action, Shooter, Sports): ").strip(),
            "Publisher": input("Editora (ej. Nintendo, Activision, Ubisoft): ").strip(),
        }
        respuesta = servicio.predecir(peticion)
        respuestas.append(respuesta)
        if respuesta["estado"] == "ok":
            barra = "█" * round(respuesta["probabilidad"] * 20)
            print(f"→ {respuesta['prediccion']} · {respuesta['probabilidad']:.0%} {barra}")
    return respuestas


def mostrar_resumen(respuestas: list[dict[str, Any]]) -> None:
    """Tabla final con todas las respuestas."""
    tabla = pd.DataFrame(respuestas)
    columnas = [c for c in ["Name", "estado", "probabilidad", "prediccion", "latencia_ms", "detalle"] if c in tabla]
    print("\n" + "=" * 70 + "\nRESUMEN DE PETICIONES\n" + "=" * 70)
    print(tabla[columnas].fillna("").to_string(index=False))
    print(f"\nAtendidas: {len(tabla)} · correctas: {(tabla['estado'] == 'ok').sum()} · "
          f"con error: {(tabla['estado'] == 'error').sum()} · log: {RUTA_LOG}")


def mostrar_grafica(respuestas: list[dict[str, Any]], ruta_salida: Path) -> None:
    """Muestra dos graficas con las predicciones y guarda una copia HTML."""
    try:
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        raise SystemExit("Para las graficas instala Plotly: python -m pip install plotly")

    tabla = pd.DataFrame(respuestas)
    correctas = tabla[tabla["estado"] == "ok"].copy()
    if correctas.empty:
        print("\nNo hay predicciones validas para representar.")
        return

    correctas["probabilidad_porcentaje"] = correctas["probabilidad"] * 100
    figura_barras = px.bar(
        correctas,
        x="probabilidad_porcentaje",
        y="Name",
        orientation="h",
        color="prediccion",
        text=correctas["probabilidad_porcentaje"].round(1).astype(str) + "%",
        labels={
            "probabilidad_porcentaje": "Probabilidad (%)",
            "Name": "Juego",
        },
    )

    conteo = correctas["prediccion"].value_counts().reset_index()
    conteo.columns = ["prediccion", "juegos"]
    figura_tarta = px.pie(
        conteo,
        names="prediccion",
        values="juegos",
        hole=0.45,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        subplot_titles=("Probabilidad de superventas por juego", "Distribucion de las predicciones"),
    )
    for traza in figura_barras.data:
        fig.add_trace(traza, row=1, col=1)
    for traza in figura_tarta.data:
        fig.add_trace(traza, row=1, col=2)

    fig.update_xaxes(title_text="Probabilidad (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Juego", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        title="Predicciones del modelo de superventas",
        height=600,
        showlegend=True,
    )
    fig.write_html(ruta_salida, include_plotlyjs="cdn")
    print(f"\nGrafica guardada en: {ruta_salida}")
    fig.show()


def modo_evaluacion(servicio: ServicioSuperventas, ruta_csv: Path) -> None:
    """Genera un informe HTML con todas las graficas del notebook."""
    from html import escape
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split

    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontro el CSV de evaluacion: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    columnas = [*servicio.variables, "Global_Sales"]
    faltan = [columna for columna in columnas if columna not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas necesarias en {ruta_csv.name}: {faltan}")

    columnas_correlacion = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    faltan_correlacion = [columna for columna in columnas_correlacion if columna not in df.columns]
    if faltan_correlacion:
        raise ValueError(f"Faltan columnas necesarias para las graficas: {faltan_correlacion}")

    df = df.copy()
    df["superventas"] = (df["Global_Sales"] >= 1).astype(int)
    conteo_superventas = (
        df["superventas"]
        .map({0: "No (< 1 M)", 1: "Sí (≥ 1 M)"})
        .value_counts()
        .reindex(["No (< 1 M)", "Sí (≥ 1 M)"], fill_value=0)
    )
    por_ano = (
        df.dropna(subset=["Year"])
        .query("Year <= 2016")
        .groupby("Year")
        .agg(juegos=("Name", "size"), tasa_superventas=("superventas", "mean"))
        .reset_index()
    )
    por_genero = (
        df.groupby("Genre")
        .agg(juegos=("Name", "size"), tasa_superventas=("superventas", "mean"))
        .sort_values("tasa_superventas")
        .reset_index()
    )
    por_plataforma = (
        df.groupby("Platform")
        .agg(juegos=("Name", "size"), tasa_superventas=("superventas", "mean"))
        .query("juegos >= 100")
        .sort_values("tasa_superventas")
        .reset_index()
    )
    por_editora = (
        df.groupby("Publisher")
        .agg(juegos=("Name", "size"), tasa_superventas=("superventas", "mean"))
        .nlargest(15, "juegos")
        .sort_values("tasa_superventas")
        .reset_index()
    )
    if por_genero.empty or por_plataforma.empty or por_ano.empty or por_editora.empty:
        raise ValueError("No hay suficientes datos para generar todas las graficas del informe.")

    X = df[servicio.variables]
    y = df["superventas"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    probabilidades = servicio.modelo.predict_proba(X_test)[:, 1]
    predicciones = (probabilidades >= servicio.umbral).astype(int)
    matriz_confusion = confusion_matrix(y_test, predicciones, labels=[0, 1])
    falsos_positivos, verdaderos_positivos, _ = roc_curve(y_test, probabilidades)
    auc = roc_auc_score(y_test, probabilidades)
    precision = precision_score(y_test, predicciones, zero_division=0)
    recall = recall_score(y_test, predicciones, zero_division=0)
    f1 = f1_score(y_test, predicciones, zero_division=0)
    cantidad_variables = len(
        servicio.modelo.named_steps[
            next(
                nombre for nombre in ("preprocesador", "preprocessor")
                if nombre in servicio.modelo.named_steps
            )
        ].get_feature_names_out()
    )
    tasa_superventas = float(y.mean())
    nombre_modelo = type(servicio.modelo.steps[-1][1]).__name__
    nombre_presentacion = {
        "LogisticRegression": "Regresión logística",
    }.get(nombre_modelo, nombre_modelo)

    graficas = []

    conteo_df = conteo_superventas.rename_axis("superventas").reset_index(name="juegos")
    conteo_df["porcentaje"] = conteo_df["juegos"] / conteo_df["juegos"].sum()
    fig = px.bar(
        conteo_df,
        x="superventas",
        y="juegos",
        color="superventas",
        color_discrete_map={"No (< 1 M)": "#8a8f98", "Sí (≥ 1 M)": "#eb6834"},
        text=[f"{n:,} ({p:.1%})" for n, p in zip(conteo_df["juegos"], conteo_df["porcentaje"])],
        title="¿Cuántos juegos son superventas?",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(template="plotly_white", showlegend=False, xaxis_title=None, yaxis_title="Número de juegos")
    graficas.append(("Proporción de superventas", "Se considera superventas un juego con ventas globales iguales o superiores a un millón. La clase está desbalanceada: solo una parte de los títulos alcanza ese nivel.", fig))

    fig = px.bar(por_ano, x="Year", y="juegos", title="Juegos lanzados por año", labels={"Year": "Año", "juegos": "Juegos lanzados"})
    fig.update_traces(marker_color="#2a78d6")
    fig.update_layout(template="plotly_white")
    graficas.append(("Lanzamientos por año", "Muestra el número de juegos registrados para cada año. Los años posteriores a 2016 se excluyen para evitar años incompletos en este conjunto histórico.", fig))

    fig = px.line(
        por_ano,
        x="Year",
        y="tasa_superventas",
        markers=True,
        title="Porcentaje de superventas entre los juegos de cada año",
        labels={"Year": "Año", "tasa_superventas": "% de superventas"},
    )
    fig.update_traces(line_color="#eb6834")
    fig.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    graficas.append(("Tasa de superventas por año", "La línea compara la proporción de superventas anual, no el volumen absoluto de ventas.", fig))

    for tabla, categoria, titulo, nombre_categoria in (
        (por_genero, "Genre", "Porcentaje de superventas por género", "género"),
        (por_plataforma, "Platform", "Porcentaje de superventas por plataforma", "plataforma"),
        (por_editora, "Publisher", "Porcentaje de superventas de las 15 editoras con más juegos", "editora"),
    ):
        fig = px.bar(
            tabla,
            x="tasa_superventas",
            y=categoria,
            orientation="h",
            text="tasa_superventas",
            custom_data=["juegos"],
            title=titulo,
            labels={"tasa_superventas": "% de superventas", categoria: nombre_categoria.capitalize()},
        )
        fig.update_traces(
            marker_color="#2a78d6",
            texttemplate="%{text:.1%}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Superventas: %{x:.1%}<br>Juegos: %{customdata[0]:,}<extra></extra>",
        )
        fig.update_layout(template="plotly_white", xaxis_tickformat=".0%", height=560)
        if categoria == "Platform":
            fig.update_layout(title=f"{titulo} (plataformas con 100 juegos o más)", height=650)
        graficas.append((titulo, f"Cada barra muestra la tasa de superventas por {nombre_categoria}; al pasar el cursor se ve el número de juegos usado para calcularla.", fig))

    correlacion = df[columnas_correlacion].corr().round(2)
    fig = px.imshow(
        correlacion,
        text_auto=True,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        title="Correlación entre las ventas por región",
    )
    fig.update_layout(template="plotly_white", height=500)
    graficas.append(("Correlación de ventas regionales", "Las ventas regionales están relacionadas con las ventas globales. No se usan como variables predictoras porque contienen información que solo se conoce después de vender el juego (fuga de información).", fig))

    etiquetas = ["No superventas", "Superventas"]
    fig = px.imshow(
        matriz_confusion,
        x=etiquetas,
        y=etiquetas,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicho", "y": "Real", "color": "Juegos"},
        title="Matriz de confusión (test)",
    )
    fig.update_layout(template="plotly_white", height=450)
    graficas.append(("Matriz de confusión", "Compara la clase real con la predicción. La diagonal son aciertos; las celdas fuera de ella son errores.", fig))

    fig = px.area(
        x=falsos_positivos,
        y=verdaderos_positivos,
        title=f"Curva ROC (AUC = {auc:.3f})",
        labels={"x": "Tasa de falsos positivos", "y": "Tasa de verdaderos positivos (recall)"},
    )
    fig.update_traces(line_color="#2a78d6")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#8a8f98"})
    fig.update_layout(template="plotly_white", height=480, width=720)
    graficas.append(("Curva ROC", "Resume el equilibrio entre detectar superventas y producir falsas alarmas para distintos umbrales. Un AUC más próximo a 1 indica mejor capacidad de discriminación.", fig))

    nombres = servicio.modelo.named_steps[
        next(nombre for nombre in ("preprocesador", "preprocessor") if nombre in servicio.modelo.named_steps)
    ].get_feature_names_out()
    coeficientes = pd.Series(
        servicio.modelo.steps[-1][1].coef_[0],
        index=nombres,
    )
    coeficientes.index = (
        coeficientes.index.str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("_infrequent_sklearn", " (poco frecuentes)", regex=False)
    )
    extremos = pd.concat([coeficientes.nsmallest(10), coeficientes.nlargest(10)]).sort_values().reset_index()
    extremos.columns = ["variable", "coeficiente"]
    extremos["efecto"] = np.where(extremos["coeficiente"] > 0, "Aumenta la probabilidad", "Reduce la probabilidad")
    fig = px.bar(
        extremos,
        x="coeficiente",
        y="variable",
        orientation="h",
        color="efecto",
        color_discrete_map={"Aumenta la probabilidad": "#2a78d6", "Reduce la probabilidad": "#eb6834"},
        title="Variables con más influencia en la predicción",
        labels={"coeficiente": "Coeficiente de la regresión logística", "variable": ""},
    )
    fig.update_layout(template="plotly_white", height=650, legend_title_text="")
    graficas.append(("Influencia de las variables", "Se muestran los diez coeficientes más positivos y los diez más negativos. En la regresión logística, el signo indica la dirección de la asociación aprendida, no una causalidad.", fig))

    ruta_salida = CARPETA / "evaluacion_modelo.html"
    secciones_graficas = []
    for indice, (titulo, descripcion, figura) in enumerate(graficas):
        grafica_html = figura.to_html(
            full_html=False,
            include_plotlyjs="cdn" if indice == 0 else False,
            config={"responsive": True},
        )
        secciones_graficas.append(
            f'<section class="chart-panel"><h2>{escape(titulo)}</h2>'
            f'<p>{escape(descripcion)}</p><div class="chart">{grafica_html}</div></section>'
        )
    bloque_graficas = "\n".join(secciones_graficas)
    explicacion = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modelo de predicción de videojuegos superventas</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; color: #172033; background: #f3f6fb; }}
    body {{ margin: 0; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 38px 28px 56px; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 4vw, 3rem); color: #12345a; }}
    h2 {{ margin: 30px 0 10px; color: #12345a; }}
    p {{ line-height: 1.65; }}
    .intro {{ max-width: 920px; font-size: 1.08rem; color: #43536b; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin: 24px 0; }}
    .card, .panel {{ background: white; border: 1px solid #e1e8f0; border-radius: 12px; padding: 18px 20px; box-shadow: 0 3px 12px #16324f0a; }}
    .card strong {{ display: block; color: #12345a; font-size: 1.65rem; margin-bottom: 5px; }}
    .card span, .muted {{ color: #63738a; }}
    .columns {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .panel h2 {{ margin-top: 0; }}
    ul {{ padding-left: 21px; line-height: 1.75; }}
    .chart {{ margin-top: 24px; background: white; border-radius: 12px; padding: 8px; border: 1px solid #e1e8f0; }}
    footer {{ margin-top: 18px; color: #63738a; font-size: .9rem; }}
    @media (max-width: 600px) {{ main {{ padding: 24px 14px 36px; }} }}
  </style>
</head>
<body>
<main>
  <h1>¿Qué hace que un videojuego sea superventas?</h1>
  <p class="intro">Este proyecto utiliza un modelo de aprendizaje automático para estimar si un videojuego alcanzará
  al menos <strong>un millón de unidades vendidas</strong>. Las gráficas comparan la proporción de superventas
  dentro de cada género y plataforma; no representan una garantía de éxito ni una relación causal.</p>

  <section class="cards" aria-label="Resumen del conjunto de datos y el modelo">
    <div class="card"><strong>{len(df):,}</strong><span>juegos analizados</span></div>
    <div class="card"><strong>{tasa_superventas:.1%}</strong><span>son superventas en los datos</span></div>
    <div class="card"><strong>{len(servicio.variables)}</strong><span>variables de entrada</span></div>
    <div class="card"><strong>{cantidad_variables}</strong><span>características tras el preprocesamiento</span></div>
    <div class="card"><strong>{len(X) - len(X_test):,} / {len(X_test):,}</strong><span>juegos para entrenamiento / prueba</span></div>
    <div class="card"><strong>{servicio.umbral:.0%}</strong><span>umbral de clasificación</span></div>
  </section>

  <section class="columns">
    <article class="panel">
      <h2>Cómo funciona el modelo</h2>
      <p>Se utiliza una <strong>{escape(nombre_presentacion)}</strong> dentro de un pipeline, que prepara los datos
      y calcula la probabilidad estimada de superventas.</p>
      <ul>
        <li><strong>Año (Year):</strong> se completan valores ausentes con la mediana y se estandariza.</li>
        <li><strong>Plataforma (Platform), género (Genre) y editora (Publisher):</strong> se completan ausencias con
        la categoría más frecuente y se convierten en variables numéricas.</li>
        <li>Las categorías con menos de 30 ejemplos se agrupan como poco frecuentes; categorías nuevas se gestionan
        sin detener la predicción.</li>
        <li>La etiqueta «superventas» significa ventas globales de al menos 1 millón.</li>
      </ul>
    </article>
    <article class="panel">
      <h2>Datos y evaluación</h2>
      <ul>
        <li>El conjunto se divide en <strong>80 % para entrenamiento y 20 % para prueba</strong>, manteniendo la
        proporción entre clases.</li>
        <li>La ponderación de clases ayuda a que el modelo preste atención a la clase minoritaria.</li>
        <li><strong>AUC en prueba: {auc:.3f}</strong>. Mide la capacidad de ordenar un superventas por encima de
        un juego que no lo es (0,5 equivale al azar; 1,0 es una separación perfecta).</li>
        <li><strong>Recall de superventas: {recall:.1%}</strong>; identifica esta proporción de los superventas reales.</li>
        <li><strong>Precisión de superventas: {precision:.1%}</strong>; de los juegos marcados, esta proporción sí
        alcanza el umbral.</li>
        <li><strong>F1 de superventas: {f1:.1%}</strong>; combina precisión y recall en una sola medida.</li>
      </ul>
      <p class="muted">Evaluación calculada con una partición de prueba reproducible del archivo
      <strong>{escape(ruta_csv.name)}</strong>.</p>
    </article>
  </section>

  <h2>Todas las visualizaciones del análisis</h2>
  <p class="intro">Explora las gráficas interactivas del análisis exploratorio y del rendimiento del modelo.
  En las gráficas por categoría puedes pasar el cursor sobre las barras para consultar sus valores.</p>
  {bloque_graficas}
  <footer>Las tasas describen este conjunto de datos histórico y no deben interpretarse como una predicción causal
  ni como una garantía de ventas futuras.</footer>
</main>
</body>
</html>"""
    ruta_salida.write_text(explicacion, encoding="utf-8")
    print(f"Graficas guardadas en: {ruta_salida}")
    import webbrowser

    webbrowser.open(ruta_salida.resolve().as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulacion de produccion del modelo de superventas")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--interactivo", action="store_true", help="Introducir los datos por teclado")
    grupo.add_argument("--entrada", type=Path, help="CSV con los juegos a predecir")
    grupo.add_argument("--evaluar", action="store_true", help="Generar un informe HTML con todas las graficas del notebook")
    parser.add_argument("--modelo", type=Path, default=RUTA_MODELO, help="Ruta del .pkl")
    parser.add_argument("--graficas", action="store_true", help="Mostrar dos graficas de las predicciones")
    args = parser.parse_args()

    logger = configurar_log()
    try:
        servicio = ServicioSuperventas(args.modelo, logger)
    except FileNotFoundError as error:
        logger.error(str(error))
        raise SystemExit(1)

    if args.evaluar:
        modo_evaluacion(servicio, CARPETA / "vgsales.csv")
    elif args.interactivo:
        respuestas = modo_interactivo(servicio)
        ruta_grafica = CARPETA / "predicciones_interactivas.html"
    elif args.entrada:
        respuestas = modo_lotes(servicio, args.entrada)
        ruta_grafica = args.entrada.with_name(args.entrada.stem + "_graficas.html")
    else:
        respuestas = modo_ejemplo(servicio)
        ruta_grafica = CARPETA / "predicciones_ejemplo.html"
    if args.graficas and not args.evaluar:
        mostrar_grafica(respuestas, ruta_grafica)
    logger.info("Servicio detenido.")


if __name__ == "__main__":
    main()
