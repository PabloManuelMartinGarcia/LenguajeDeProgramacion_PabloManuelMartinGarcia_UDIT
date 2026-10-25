
#Ejercicio 1 

#**1. Indexado/Slicing**  
#Dada `s = "programación"`, obtén:
#- Las 4 primeras letras.



s= "programación"
s[0:4] 

#- Las 3 últimas letras.

s= "programación"
s[-3:] 

#- La cadena invertida.
s[::-1]

#- Cada 2 caracteres empezando en el índice 1.
s[1:10:2]


#**2. Membership**  

#Con `palabras = ["ia", "python", "datos"]`:

palabras = ["ia", "python", "datos"]

#- Comprueba si `"py"` está en `"python"`.

s.find("py") != 1

#- Comprueba si `"ml"` está en la lista.

palabras,"ml" in palabras

#- Convierte la lista a `set` y comprueba la pertenencia de `"datos"`.

set(palabras)


#**3. `len`**  

#¿Qué devuelve `len([[], [1,2], "ab"])`? Explica por qué.

len([[], [1,2], "ab"]) 

# Devuelve 3 porque la función `len()` cuenta el número de elementos en la lista. 
# En este caso, la lista contiene tres elementos: una lista vacía `[]`, una lista con dos elementos `[1,2]`, 
# y una cadena `"ab"`. Por lo tanto, el resultado es 3.

#**4. Cadenas**  

#Para `s = "  Python,3.12  "`:
#Límpiala a `"Python,3.12"`.
s="  Python,3.12  "
# Sepárala por `","` y conviértelo en `("Python", "3.12")`.
s.split(",")
# Únelas con `" - "` para obtener `"Python - 3.12"`.
"-".join(s.split(","))

#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 2

#1. Escribe un programa que pida un número al usuario y diga si es positivo, negativo o cero.  

for numero in [int(input("Introduce un número: "))]:
    if numero > 0:
        print("El número es positivo.")
    elif numero < 0:
        print("El número es negativo.")
    else:
        print("El número es cero.")

#2. Recorre una lista de nombres e imprime solo aquellos que tengan más de 4 letras.  

nombres = ["Ana", "Carlos", "Elena", "Francisco", "Gloria"]
for nombre in nombres:
    if len(nombre) > 4:
        print(nombre)

#3. Implementa un `while` que vaya contando desde 1 hasta que el usuario escriba `"stop"`.  

contador = 1
while True:
    print(contador)
    entrada = input("Escribe 'stop' para detener: ")
    if entrada == "stop":
        break
    contador += 1

#4. Escribe un bucle que recorra una lista de enteros y construya otra lista con solo los valores pares.

numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pares = []
for n in numeros:
    if n % 2 == 0:
        pares.append(n)
print(pares)

#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 3

#1. Define una función `es_par` que reciba un número y devuelva `True` si es par, `False` si no.  

es_par = lambda x: x % 2 == 0
print(es_par(4))  # Devuelve True

#2. Crea una función `media` que acepte cualquier cantidad de números (`*args`) y calcule la media.  

media = lambda *args: sum(args) / len(args) if args else 0
print(media(1, 2, 3, 4, 5))  # Devuelve 3.0 

#3. Usa `map` con una función `lambda` para convertir una lista de temperaturas en grados Celsius a Fahrenheit. 

fahrenheit = list(map(lambda c: (c * 9/5) + 32, [0, 20, 37, 100]))  # Convierte a Fahrenheit

#4. Usa `filter` para obtener de una lista de nombres aquellos que empiecen con la letra `"A"`.  

filter_nombres = list(filter(lambda nombre: nombre.startswith("A"), ["Ana", "Carlos", "Alberto", "Elena"])) 

#5. Usa `zip` para combinar dos listas: nombres y notas, y luego recórrelas con un `for` para mostrar `"Alumno X → Nota Y"`.  

zip_nombres_notas = zip(["Ana", "Carlos", "Elena"], [8, 7, 9])
for nombre, nota in zip_nombres_notas:
    print(f"Alumno {nombre} → Nota {nota}")

#6. Recorre una lista de ciudades con `enumerate` y muestra la posición junto con el nombre.

ciudades = ["Madrid", "Barcelona", "Valencia", "Sevilla"]
for i, ciudad in enumerate(ciudades):
    print(f"Posición {i}: {ciudad}")

#7. Dada una lista de palabras, usa **`map`** para pasarlas a *title case* y **`filter`** para quedarte solo con aquellas de longitud ≥ 5.

palabras = ["hola", "mundo", "python", "programación"]
title_case = list(map(str.title, palabras))
filter_title_case = list(filter(lambda palabra: len(palabra) >= 5, title_case))
print(filter_title_case)

#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 4

#1. Declara tres variables con anotaciones de tipo:  
  #- Un entero para la edad.  
   #- Una lista de números reales con precios.  
   #- Una tupla de dos enteros representando coordenadas.  

edad: int = 25
precios: list[float] = [19.99, 29.99, 9.99]
coordenadas: tuple[int, int] = (10, 20)

#2. Escribe una función `es_mayor_de_edad(edad: int) -> bool` que devuelva `True` si la edad es mayor o igual a 18.  

def es_mayor_de_edad(edad: int) -> bool:
    return edad >= 18
print(es_mayor_de_edad(20))  # Devuelve True

#3. Implementa una función `concatena(a: str, b: Optional[str]) -> str` que devuelva la concatenación de `a` y `b`, usando cadena vacía si `b` es `None`.  

def concatena(a: str, b: str | None) -> str:
    return a + (b if b is not None else "") 

#4. Define una función `promedio(valores: List[float]) -> float` que calcule la media de una lista de números reales.  

def promedio(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0
    
#5. Declara una función `procesar(valor: Union[int, str]) -> str` que: 
 #- Si recibe un entero, devuelva su cuadrado como cadena.  
# - Si recibe un string, lo devuelva en mayúsculas.

def procesar(valor: int | str) -> str:
    if isinstance(valor, int):
        return str(valor ** 2)
    elif isinstance(valor, str):
        return valor.upper()
    else:
        raise ValueError("El valor debe ser un entero o una cadena.")

#6. Anota los tipos de la función y corrige la implementación si es necesario:

def suma_mixta(a: float, b: float) -> float:
    return a + b

#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 5

### 📝 **Ejercicios**

#1. Escribe un docstring en estilo **Google** para una función `es_par(n: int) -> bool`
   #que devuelva `True` si el número es par, `False` si no.

def es_par(n: int) -> bool:
    """Determina si un número es par.

    Args:
        n (int): El número a evaluar.

    Returns:
        bool: `True` si el número es par, `False` en caso contrario.
    """
    return n % 2 == 0

#2. Documenta en estilo **Google** una función `area_circulo(radio: float) -> float`.

def area_circulo(radio: float) -> float:
    """Calcula el área de un círculo dado su radio.

    Args:
        radio (float): El radio del círculo.

    Returns:
        float: El área del círculo.
    """
    import math
    return math.pi * radio ** 2

#3. Crea una función `factorial(n: int) -> int` con su docstring en estilo Google.

def factorial(n: int) -> int:
    """Calcula el factorial de un número entero no negativo.

    Args:
        n (int): El número entero no negativo del cual se desea calcular el factorial.

    Returns:
        int: El factorial de `n`.

    Raises:
        ValueError: Si `n` es negativo.
    """
    if n < 0:
        raise ValueError("El número debe ser no negativo.")
    if n == 0 or n == 1:
        return 1
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado
#4. Genera un docstring para un módulo que implemente funciones matemáticas básicas.

"""
Módulo de funciones matemáticas básicas.

Este módulo proporciona funciones para realizar operaciones matemáticas simples.

"""

#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 6


#1. Escribe una función que abra un fichero cuyo nombre se pasa como argumento:  
#- Si **no existe**, capture la excepción y muestre un mensaje.  
#- Si existe, devuelva el número de líneas que contiene.

def contar_lineas(path: str) -> int:
    """Cuenta el número de líneas en un fichero.

    Args:
        path (str): La ruta del fichero.

    Returns:
        int: El número de líneas en el fichero.
    """
    try:
        with open(path, 'r') as f:
            return sum(1 for line in f)
    except FileNotFoundError:
        print("El fichero no existe.")
        return 0

#2. Escribe una función `lee_float(path)` que intente leer un número en coma flotante desde un fichero.
#- Si el fichero **no existe**, devuelve `None`.
#- Si el contenido **no es numérico**, lanza un `ValueError`.
#- En caso contrario, devuelve el `float` leído.

def lee_float(path: str) -> float | None:
    """Lee un número en coma flotante desde un fichero.

    Args:
        path (str): La ruta del fichero.

    Returns:
        float | None: El número leído como float, o None si el fichero no existe.
         """
    try:
        with open(path, 'r') as f:
            contenido = f.read().strip()
            return float(contenido)
    except FileNotFoundError:
        return None
    except ValueError:
        raise ValueError("El contenido del fichero no es numérico.")
    
#------------------------------------------------------------------------------------------------------------------------

#Ejercicio 7

#1. Crea un set comprehension con los números entre 0 y 20 que sean múltiplos de 3..  

multiples_de_3 = {x for x in range(21) if x % 3 == 0} 

#2. A partir de la lista `frases = ["hola mundo", "python es genial", "list comprehensions"]`,
#crea una lista con todas las palabras en mayúsculas, usando una comprensión anidada.. 

frases = ["hola mundo", "python es genial", "list comprehensions"]
palabras_mayusculas = [palabra.upper() for frase in frases for palabra in frase.split()]

#------------------------------------------------------------------------------------------------------------------------

#MINI-RETO INTEGRADOR

# Crea una función `analiza_log(path)` que lea un fichero de texto con líneas del tipo:
# ```
# INFO: usuario=ana tiempo=0.23
# ERROR: usuario=pepe tiempo=0.01
# WARNING: usuario=marta tiempo=1.23
# ```
#La función debe devolver un `dict` con:
#- El **número total** de líneas por nivel (`INFO`, `ERROR`, `WARNING`).
#- El **tiempo medio** global (promedio de todos los `tiempo=`).
#- Una lista de **usuarios únicos** en orden alfabético.
# TODO: implementa analiza_log(path)

def analiza_log(path):
    """Analiza un fichero de log y devuelve estadísticas.

    Args:
        path (str): La ruta del fichero de log.

    Returns:
        dict: Un diccionario con el número total de líneas por nivel, 
              el tiempo medio global y una lista de usuarios únicos en orden alfabético.
    """
    from collections import defaultdict
    niveles = defaultdict(int)
    tiempos = []
    usuarios = set()

    try:
        with open(path, 'r') as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) < 3:
                    continue  # Saltar líneas mal formateadas
                nivel = partes[0].rstrip(':')
                usuario = partes[1].split('=')[1]
                tiempo = float(partes[2].split('=')[1])

                niveles[nivel] += 1
                tiempos.append(tiempo)
                usuarios.add(usuario)

        tiempo_medio = sum(tiempos) / len(tiempos) if tiempos else 0.0
        return {
            "niveles": dict(niveles),
            "tiempo_medio": tiempo_medio,
            "usuarios_unicos": sorted(usuarios)
        }
    except FileNotFoundError:
        print("El fichero no existe.")
        return {}