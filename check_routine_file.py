import json

def validar_json(ruta_archivo):
    try:
        with open(ruta_archivo, 'r') as archivo:
            json.load(archivo)
        print("El JSON está bien formado.")
    except json.JSONDecodeError as error:
        print(f"JSON mal formado: {error}")
    except FileNotFoundError:
        print("El archivo no existe.")

# Uso
ruta = "test_routines.routine"  # Cambia esto por la ruta de tu archivo JSON
validar_json(ruta)