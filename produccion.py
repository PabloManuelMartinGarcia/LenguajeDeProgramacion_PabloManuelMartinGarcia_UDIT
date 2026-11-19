"""Simulacion de produccion del modelo de superventas.

Carga una sola vez el modelo entrenado (``modelo_superventas.pkl``), como haria
un servicio al arrancar, y despues atiende "peticiones": valida los datos de
entrada, predice y registra cada respuesta en un log.

Uso:

    python produccion.py                       # simula una tanda de peticiones de ejemplo
    python produccion.py --interactivo         # escribes tu los datos del juego
    python produccion.py --entrada juegos.csv  # prediccion por lotes de un CSV
    python produccion.py --evaluar             # evalua el .pkl y muestra dos graficas

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
    """Carga el modelo y muestra las tasas de superventas por genero y plataforma."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if not ruta_csv.is_file():
        raise FileNotFoundError(f"No se encontro el CSV de evaluacion: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    columnas = [*servicio.variables, "Global_Sales"]
    faltan = [columna for columna in columnas if columna not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas necesarias en {ruta_csv.name}: {faltan}")

    df = df.dropna(subset=["Genre", "Platform"]).copy()
    df["superventas"] = (df["Global_Sales"] >= 1).astype(int)
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
    if por_genero.empty or por_plataforma.empty:
        raise ValueError("No hay suficientes datos para calcular las tasas por genero y plataforma.")

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Porcentaje de superventas por género", "Porcentaje de superventas por plataforma"),
    )
    for tabla, categoria, col in (
        (por_genero, "Genre", 1),
        (por_plataforma, "Platform", 2),
    ):
        fig.add_trace(
            go.Bar(
                x=tabla["tasa_superventas"],
                y=tabla[categoria],
                orientation="h",
                marker_color="#2a78d6",
                text=tabla["tasa_superventas"],
                texttemplate="%{text:.1%}",
                textposition="outside",
                customdata=tabla[["juegos"]],
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    "%{y}<br>Superventas: %{x:.1%}<br>"
                    "Juegos: %{customdata[0]:,}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Porcentaje de superventas por género y plataforma",
        template="plotly_white",
        height=600,
        width=1200,
        showlegend=False,
    )
    fig.update_xaxes(title_text="% de superventas", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="% de superventas", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Género", row=1, col=1)
    fig.update_yaxes(title_text="Plataforma", row=1, col=2)
    ruta_salida = CARPETA / "evaluacion_modelo.html"
    fig.write_html(ruta_salida, include_plotlyjs="cdn")
    print(f"Graficas guardadas en: {ruta_salida}")
    fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulacion de produccion del modelo de superventas")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--interactivo", action="store_true", help="Introducir los datos por teclado")
    grupo.add_argument("--entrada", type=Path, help="CSV con los juegos a predecir")
    grupo.add_argument("--evaluar", action="store_true", help="Cargar el .pkl y mostrar las dos primeras graficas del EDA")
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
