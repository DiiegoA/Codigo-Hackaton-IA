#!/usr/bin/env python
# coding: utf-8

# # FASE 1: Ingesta, Auditoría y Muro de Hierro
# Aquí los datos entran, se limpian sintácticamente y se aíslan para evitar que el modelo vea el futuro.

# In[25]:


# ==========================================
# 0. Blindaje de Dependencias (Multi-Motor)
# ==========================================
import os
import csv
import time
import pandas as pd
from typing import Tuple, Any, Optional
from IPython.display import display

# Intentamos cargar el motor de ultra-alta velocidad (Polars)
try:
    import polars as pl
    MOTOR_PRINCIPAL = 'polars'
except ImportError:
    MOTOR_PRINCIPAL = 'pandas'
    print("⚠️ Aviso: 'polars' no detectado. Usando 'pandas' (Motor C) como respaldo de emergencia.")


def ingesta_acelerada_multicore(
    ruta_archivo: str,
    convertir_a_pandas_mutable: bool = True
) -> Tuple[Optional[Any], int]:
    """
    [FASE 1 - Paso 1.1] Ingesta Acelerada Multi-core (Nivel Producción 2026).
    - Motor: Usa Polars (Rust) para leer millones de filas usando todos los hilos del CPU.
    - Inteligencia Escalonada: Sniffer primario + Escáner de frecuencia de respaldo para delimitadores rebeldes.
    - Tolerancia a Encodings (NUEVO): Ignora caracteres corruptos al detectar el separador.
    - Mutabilidad: Extrae los datos al backend nativo de NumPy permitiendo cirugía de datos posterior.
    """
    # ==========================================
    # 1. Cláusulas de Guarda (Seguridad del File System)
    # ==========================================
    if not ruta_archivo or not isinstance(ruta_archivo, str):
        print("🛑 Error Crítico: Ruta de archivo inválida o nula.")
        return None, 0

    if not os.path.exists(ruta_archivo):
        print(f"🛑 Error Crítico: El archivo '{ruta_archivo}' no fue encontrado en el sistema.")
        return None, 0

    print("=== 🚀 FASE 1.1: Ingesta Acelerada Estructural ===")

    # ==========================================
    # 2. Inteligencia de Detección de Separador (Escudo Doble + Tolerancia a Fallos)
    # ==========================================
    separador_detectado = ','  # Default absoluto
    try:
        # 🔧 FIX MLOPS: errors='replace' evita que el código crashee si hay bytes corruptos (ej. 0xa0)
        with open(ruta_archivo, 'r', encoding='utf-8', errors='replace') as archivo:
            # Leemos solo un bloque minúsculo para no ahogar la RAM
            muestra = archivo.read(10240) 

            try:
                # INTENTO 1: Motor Sniffer oficial de Python
                separador_detectado = csv.Sniffer().sniff(muestra).delimiter

                # Manejo especial: A veces el sniffer confunde letras normales con separadores en textos sucios
                if separador_detectado.isalnum():
                    raise ValueError("Sniffer detectó una letra/número como separador. Activando respaldo.")

                print(f" 🔍 Heurística Principal: Separador '{separador_detectado}' detectado por Sniffer.")

            except Exception:
                # INTENTO 2: Escáner de Frecuencia (El Fallback Inteligente)
                separadores_candidatos = [',', ';', '\t', '|']
                lineas = muestra.strip().split('\n')[:10] # Analizamos las primeras 10 líneas

                # Contamos cuántas veces aparece cada candidato en la muestra
                conteos = {sep: sum(linea.count(sep) for linea in lineas) for sep in separadores_candidatos}
                mejor_candidato = max(conteos, key=conteos.get)

                if conteos[mejor_candidato] > 0:
                    separador_detectado = mejor_candidato
                    if separador_detectado == '\t':
                        print(" 🛡️ Heurística de Respaldo: Sniffer falló, pero se detectó 'TABULADOR' por frecuencia.")
                    else:
                        print(f" 🛡️ Heurística de Respaldo: Sniffer falló, pero se detectó '{separador_detectado}' por frecuencia.")
                else:
                    print(" ⚠️ Alerta MLOps: Formato irreconocible o archivo de una sola columna. Forzando coma (',').")

    except Exception as e:
        print(f" 🛑 Error fatal al inspeccionar el archivo: {e}. Forzando coma (',').")

    # ==========================================
    # 3. Motor de Ingesta de Alta Velocidad
    # ==========================================
    df_resultante = None
    inicio_timer = time.time()

    try:
        if MOTOR_PRINCIPAL == 'polars':
            print(f" ⚡ Ejecutando ingesta paralela con motor POLARS (Multi-core)...")

            # Polars lee en paralelo, ignora errores de codificación (utf8-lossy) y líneas corruptas
            df_polars = pl.read_csv(
                ruta_archivo,
                separator=separador_detectado,
                ignore_errors=True,
                infer_schema_length=10000,
                encoding='utf8-lossy'
            )

            # MLOps: Convertimos a Pandas nativo (NumPy backend).
            if convertir_a_pandas_mutable:
                df_resultante = df_polars.to_pandas()
                print("    ✔️ Archivo cargado a velocidad extrema y convertido a Pandas Mutable.")
            else:
                df_resultante = df_polars
                print("    ✔️ Archivo cargado a velocidad extrema (Mantenido en Polars).")

        else:
            # Fallback a Pandas si el usuario no tiene Polars instalado
            print(f" 🐢 Ejecutando ingesta con motor PANDAS (C-Engine)...")
            df_resultante = pd.read_csv(
                ruta_archivo,
                sep=separador_detectado,
                engine='c',
                on_bad_lines='skip',
                low_memory=False
            )
            print("    ✔️ Archivo cargado mediante fallback de seguridad (Nativo Mutable).")

    except Exception as e:
        print(f"🛑 Error crítico durante la lectura del archivo: {e}")
        return None, 0

    tiempo_total = time.time() - inicio_timer

    # ==========================================
    # 4. Snapshot de Memoria y Reporte UI
    # ==========================================
    filas, columnas = df_resultante.shape

    # Cálculo de memoria seguro (Dependiendo si es Pandas o Polars)
    if isinstance(df_resultante, pd.DataFrame):
        memoria_mb = df_resultante.memory_usage(deep=True).sum() / (1024 ** 2)
    else:
        memoria_mb = df_resultante.estimated_size() / (1024 ** 2)

    print(f"\n📊 Diagnóstico de Ingesta:")
    print(f"  ⏱️ Tiempo de lectura : {tiempo_total:.4f} segundos")
    print(f"  📐 Dimensiones       : {filas:,} filas x {columnas} columnas")
    print(f"  💾 Consumo de RAM    : {memoria_mb:.2f} MB")

    print("\n--- 👁️ Radiografía de Estructura Inicial (Primeras 3 filas) ---")
    display(df_resultante.head(3))

    return df_resultante, filas


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
# IMPORTANTE: Si no tienes Polars, instálalo en una celda arriba con: !pip install polars

ruta_dataset = '1_Data_Recordings.csv'  # Cambia esto por tu archivo real

try:
    df_crudo, total_filas_originales = ingesta_acelerada_multicore(
        ruta_archivo=ruta_dataset,
        convertir_a_pandas_mutable=True  # 🔥 La clave para habilitar la cirugía de datos posterior
    )

except Exception as e:
    print(f"🛑 Fallo en la celda de ejecución: {e}")


# In[26]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import time
from typing import Tuple

def purgar_clones_absolutos(df_crudo: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    [FASE 1 - Paso 1.2] Purga de Clones Absolutos (Nivel Producción).
    - Inteligencia Estructural: Busca filas 100% idénticas en todas sus dimensiones.
    - Prevención de Leakage Temprano: Elimina duplicados originados por errores 
      de extracción (SQL JOINs cruzados) que inflarían el conteo estadístico.
    - MLOps: Perfila la memoria RAM liberada en el proceso.
    """
    # ==========================================
    # 1. Cláusulas de Guarda (Seguridad)
    # ==========================================
    if not isinstance(df_crudo, pd.DataFrame) or df_crudo.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df_crudo, 0

    print("=== 🧬 FASE 1.2: Purga de Clones Absolutos ===")

    inicio_timer = time.time()
    filas_originales = len(df_crudo)

    # 📸 Snapshot de memoria real (Deep)
    mem_antes = df_crudo.memory_usage(deep=True).sum() / (1024 ** 2)

    # ==========================================
    # 2. Escáner de Redundancia Total (Fuerza Bruta C)
    # ==========================================
    print(" 🔍 Escaneando la matriz en busca de espejos perfectos...")

    # duplicated() en Pandas usa tablas hash en C subyacente, es ultra-rápido
    # keep='first' marca como True a los impostores (copias) y salva al original
    mascara_clones = df_crudo.duplicated(keep='first')
    cantidad_clones = mascara_clones.sum()

    # ==========================================
    # 3. La Guillotina de Clones
    # ==========================================
    if cantidad_clones > 0:
        print(f" 🚨 ALERTA: Se detectaron {cantidad_clones:,} filas 100% idénticas.")
        print("    ↳ Acción: Ejecutando guillotina (Conservando solo el registro original)...")

        # Filtramos la matriz quedándonos solo con los que NO son clones (~mascara)
        df_sin_clones = df_crudo[~mascara_clones].copy()

        # Reseteamos el índice para que FLAML/LightGBM no colapsen por saltos numéricos
        df_sin_clones.reset_index(drop=True, inplace=True)

        # ==========================================
        # 4. Reporte Ejecutivo de Hardware
        # ==========================================
        mem_despues = df_sin_clones.memory_usage(deep=True).sum() / (1024 ** 2)
        ahorro_ram = mem_antes - mem_despues
        tiempo_total = time.time() - inicio_timer

        print(f"\n✅ Purga completada en {tiempo_total:.3f}s.")
        print(f"  📉 Filas eliminadas : {cantidad_clones:,}")
        print(f"  📊 Filas puras      : {len(df_sin_clones):,}")
        print(f"  🚀 RAM Liberada     : {ahorro_ram:.2f} MB")

    else:
        df_sin_clones = df_crudo.copy()
        tiempo_total = time.time() - inicio_timer
        print(f"\n ✔️ Matriz impecable. No se encontraron clones absolutos ({tiempo_total:.3f}s).")

    return df_sin_clones, cantidad_clones


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    # Verificamos que el DataFrame del Paso 1.1 exista en RAM
    if 'df_crudo' not in locals() and 'df_crudo' not in globals():
        raise EnvironmentError("No se encontró 'df_crudo'. Asegúrate de ejecutar la Ingesta (Paso 1.1).")

    # Ejecutamos la purga y CREAMOS el nuevo bloque de la cadena de montaje: 'df_sin_clones'
    df_sin_clones, total_clones_destruidos = purgar_clones_absolutos(df_crudo=df_crudo)

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante: {env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Purga de Clones: {e}")


# In[27]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import numpy as np
import re
from typing import Tuple, List
import time
import warnings

def ejecutar_guillotina_inteligente(
    df: pd.DataFrame,
    targets_potenciales: List[str] 
) -> Tuple[pd.DataFrame, dict]:
    """
    [FASE 1 - Paso 1.3] La Guillotina Temprana (Nivel AutoML Avanzado).
    - Batalla de Targets: Conserva el primero de la lista y destruye a sus rivales explícitos.
    - IA Anti-Leakage: Escanea TODA la matriz y decapita automáticamente variables 'tramposas'.
    - Regla de Oro: Purga filas si el Target es nulo.
    - Escáner de Degradación: Destruye columnas con >90% de nulos.
    - Escáner de Entropía V3 (NUEVO): Detecta IDs en CamelCase/PascalCase y llaves primarias desordenadas.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df, {}

    print("=== 🪓 FASE 1.3: Guillotina Temprana (IA Estructural y Migración de IDs) ===")

    inicio_timer = time.time()
    df_opt = df.copy()

    # 👑 Se extrae al ganador inmediatamente
    target_principal = targets_potenciales[0]

    reporte_operaciones = {
        'target_escogido': target_principal, 
        'filas_sin_target_eliminadas': 0, 
        'targets_secundarios_eliminados': [], 
        'fugas_datos_detectadas_ia': [], 
        'nulos_masivos_eliminados': [], 
        'constantes_eliminadas': [],
        'isomorficas_redundantes_eliminadas': [],
        'ids_migrados_al_index': []
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ==========================================
        # 1. LA BATALLA, BLINDAJE Y AUTO-LEAKAGE
        # ==========================================
        if target_principal in df_opt.columns:
            print(f" 🎯 [TARGET SELECCIONADO]: '{target_principal}' se mantendrá intacto.")
            filas_antes = len(df_opt)

            patron_regex = r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|)$|^[^a-zA-Z0-9]+$'
            mask_basura = df_opt[target_principal].astype(str).str.match(patron_regex)

            if mask_basura.any():
                df_opt.loc[mask_basura, target_principal] = np.nan

            df_opt.dropna(subset=[target_principal], inplace=True)

            filas_destruidas = filas_antes - len(df_opt)
            if filas_destruidas > 0:
                reporte_operaciones['filas_sin_target_eliminadas'] = filas_destruidas
                print(f" 🧹 [REGLA DE ORO]: {filas_destruidas} filas aniquiladas por contener Target nulo.")
        else:
            print(f" ⚠️ Alerta: El target '{target_principal}' no se encontró en la matriz.")
            return df, {}

        if len(targets_potenciales) > 1:
            targets_secundarios = targets_potenciales[1:]
            a_borrar_targets = [t for t in targets_secundarios if t in df_opt.columns]

            if a_borrar_targets:
                df_opt.drop(columns=a_borrar_targets, inplace=True)
                reporte_operaciones['targets_secundarios_eliminados'] = a_borrar_targets
                print(f" 🗑️ [BATALLA]: Decapitando rivales explícitos: {a_borrar_targets}")

        print(" 🧠 Activando Motor de Correlación para buscar Fugas de Datos ocultas...")

        trampas_descubiertas = []
        umbral_trampa = 0.85 

        rey_numerico = pd.factorize(df_opt[target_principal])[0] if df_opt[target_principal].dtype == 'object' or df_opt[target_principal].dtype == 'category' else df_opt[target_principal]

        for col in df_opt.columns:
            if col == target_principal: continue

            if pd.api.types.is_numeric_dtype(df_opt[col]) or pd.api.types.is_bool_dtype(df_opt[col]):
                mask = ~df_opt[col].isna() & (rey_numerico != -1) 
                if mask.sum() > 100: 
                    correlacion = np.abs(np.corrcoef(df_opt.loc[mask, col], rey_numerico[mask])[0, 1])

                    if correlacion >= umbral_trampa:
                        trampas_descubiertas.append(col)
                        print(f"    🚨 [Fuga Detectada]: '{col}' predice al Rey con {correlacion*100:.1f}% de exactitud. Es trampa.")

        if trampas_descubiertas:
            df_opt.drop(columns=trampas_descubiertas, inplace=True)
            reporte_operaciones['fugas_datos_detectadas_ia'] = trampas_descubiertas
            print(f"    🔪 Decapitando fugas del futuro automáticas: {trampas_descubiertas}")
        else:
            print("    ✅ La matriz parece estar libre de Fugas de Datos obvias.")

        filas_totales_actuales = len(df_opt)

        # ==========================================
        # 🚀 2. ESCÁNER DE DEGRADACIÓN (Nulos Masivos)
        # ==========================================
        umbral_nulos = 0.90 
        nulos_ratios = df_opt.isna().mean()
        a_borrar_nulos = nulos_ratios[nulos_ratios >= umbral_nulos].index.tolist()

        if target_principal in a_borrar_nulos:
            a_borrar_nulos.remove(target_principal)

        if a_borrar_nulos:
            df_opt.drop(columns=a_borrar_nulos, inplace=True)
            reporte_operaciones['nulos_masivos_eliminados'] = a_borrar_nulos
            print(f" 🕳️ [DEGRADACIÓN]: {len(a_borrar_nulos)} variables destruidas por nulos irrecuperables (>90%).")

        # ==========================================
        # 🧠 3. ESCÁNER DE ENTROPÍA V3 (IDs Inteligentes & CamelCase)
        # ==========================================
        patron_id_base = re.compile(r'(^id$|_id$|^id_|^cod_|^codigo|_codigo$|_code$|^idx$|uuid|hash|pk|cedula)', re.IGNORECASE)
        a_borrar_constantes = []
        ids_encontrados = []

        for col in df_opt.columns:
            if col == target_principal: continue

            unicos = df_opt[col].nunique(dropna=False) 

            if unicos <= 1:
                a_borrar_constantes.append(col)
                continue
            elif not pd.api.types.is_float_dtype(df_opt[col]):
                frecuencia_top = df_opt[col].value_counts(normalize=True, dropna=False).iloc[0]
                if frecuencia_top >= 0.995: 
                    a_borrar_constantes.append(col)
                    continue

            ratio_unicidad = unicos / filas_totales_actuales
            es_id = False

            # 🚀 FIX: Soporte semántico para CamelCase/PascalCase (ej. PatientId, AppointmentID)
            es_id_semantico = bool(patron_id_base.search(col))
            if not es_id_semantico and len(col) > 2:
                if col.endswith('Id') or col.endswith('ID'):
                    es_id_semantico = True

            if es_id_semantico and unicos > 10: 
                es_id = True
            elif pd.api.types.is_integer_dtype(df_opt[col]) and ratio_unicidad >= 0.95:
                # 🚀 FIX: Si un entero es >95% único, es Llave Primaria (no requiere estar ordenado).
                es_id = True
            elif pd.api.types.is_object_dtype(df_opt[col]) or pd.api.types.is_string_dtype(df_opt[col]):
                if ratio_unicidad >= 0.80:
                    longitudes = df_opt[col].dropna().astype(str).str.len()
                    if longitudes.nunique() == 1 and longitudes.iloc[0] >= 10:
                        es_id = True
            elif ratio_unicidad >= 0.99 and not pd.api.types.is_float_dtype(df_opt[col]):
                es_id = True

            if es_id:
                ids_encontrados.append(col)

        if a_borrar_constantes:
            df_opt.drop(columns=a_borrar_constantes, inplace=True)
            reporte_operaciones['constantes_eliminadas'] = a_borrar_constantes

        if ids_encontrados:
            df_opt.set_index(ids_encontrados, inplace=True)
            reporte_operaciones['ids_migrados_al_index'] = ids_encontrados
            print(f" 🔒 IDs migrados de forma segura al Index: {ids_encontrados}")

        # ==========================================
        # 4. DETECCIÓN DE ISOMORFISMO (1:1 Redundancia)
        # ==========================================
        dicc_unicos = {}
        for col in df_opt.columns:
            if col == target_principal: continue

            n_val = df_opt[col].nunique()
            if 1 < n_val <= 100:  
                dicc_unicos.setdefault(n_val, []).append(col)

        a_borrar_isomorfismo = []
        for n_val, columnas in dicc_unicos.items():
            if len(columnas) > 1: 
                for i in range(len(columnas)):
                    for j in range(i + 1, len(columnas)):
                        col_A = columnas[i]
                        col_B = columnas[j]

                        if col_A in a_borrar_isomorfismo or col_B in a_borrar_isomorfismo:
                            continue

                        combinaciones = len(df_opt[[col_A, col_B]].drop_duplicates())

                        if combinaciones == n_val:
                            if pd.api.types.is_numeric_dtype(df_opt[col_A]) and not pd.api.types.is_numeric_dtype(df_opt[col_B]):
                                a_borrar_isomorfismo.append(col_B)
                            else:
                                a_borrar_isomorfismo.append(col_A)

        if a_borrar_isomorfismo:
            df_opt.drop(columns=a_borrar_isomorfismo, inplace=True)
            reporte_operaciones['isomorficas_redundantes_eliminadas'] = a_borrar_isomorfismo

    # ==========================================
    # 5. Reporte Ejecutivo de MLOps
    # ==========================================
    tiempo_total = time.time() - inicio_timer

    columnas_destruidas = (
        len(reporte_operaciones['targets_secundarios_eliminados']) +
        len(reporte_operaciones['fugas_datos_detectadas_ia']) +
        len(reporte_operaciones['nulos_masivos_eliminados']) +
        len(reporte_operaciones['constantes_eliminadas']) +
        len(reporte_operaciones['isomorficas_redundantes_eliminadas'])
    )

    print(f"\n✅ Cirugía Estructural Completada en {tiempo_total:.4f}s:")
    print(f"  🎯 Target Definitivo         : {reporte_operaciones['target_escogido']}")
    print(f"  🔪 Filas destruidas (Basura) : {reporte_operaciones['filas_sin_target_eliminadas']}")
    print(f"  📉 Columnas destruidas       : {columnas_destruidas}")
    print(f"  🛡️ Columnas migradas a Index : {len(reporte_operaciones['ids_migrados_al_index'])}")
    print(f"  📊 Variables predictoras     : {df_opt.shape[1]}")

    return df_opt, reporte_operaciones


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'df_sin_clones' not in locals() and 'df_sin_clones' not in globals():
        raise EnvironmentError("No se encontró 'df_sin_clones'. Ejecuta el Paso 1.2 primero.")

    mis_targets = [
        'posible_frustracion',     # 👑 EL REY
        'standarized_engagement_score',       # Su rival a eliminar
    ] 

    df_purgado, reporte_guillotina = ejecutar_guillotina_inteligente(
        df=df_sin_clones,
        targets_potenciales=mis_targets 
    )

    target_ganador_fase1 = reporte_guillotina.get('target_escogido')
    print(f"\n📦 Variable guardada con éxito en memoria: target_ganador_fase1 = '{target_ganador_fase1}'")

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante: {env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en La Guillotina Temprana: {e}")


# In[28]:


# Ejecuta esto para ver el acta de defunción de tus columnas:
import json
print(json.dumps(reporte_guillotina, indent=4))


# In[29]:


# ==========================================
# 0. Blindaje de Dependencias UI
# ==========================================
from IPython.display import display, Markdown
from typing import Dict

# ==========================================
# MOTOR MLOPS: AUDITOR DE EXPLICABILIDAD
# ==========================================
def auditar_dictamen_guillotina(reporte: Dict) -> None:
    """
    [FASE 1 - Paso 1.4] Auditor del Dictamen (Explainable AI).
    - Transparencia: Traduce las eliminaciones técnicas a explicaciones de negocio.
    - Audit Trail: Justifica por qué cada columna era un riesgo matemático o estructural.
    - Clean UI: Renderiza un informe en formato Markdown ideal para Jupyter Notebooks.
    """
    if not reporte:
        print("🛑 Error: El reporte de la guillotina está vacío o no se generó correctamente.")
        return

    print("=== ⚖️ FASE 1.4: Auditoría y Justificación del Dictamen ===")

    # Textos de justificación arquitectónica
    justificaciones = {
        'targets_secundarios_eliminados': (
            "🗑️ **Targets Secundarios (Evitar la Bola de Cristal):**",
            "Se eliminaron porque dejar un target alternativo en la matriz de entrenamiento causa una 'Fuga del Futuro'. "
            "El modelo aprendería a predecir el resultado usando la respuesta de su rival, lo cual es imposible en el mundo real."
        ),
        'fugas_datos_detectadas_ia': (
            "🚨 **Fugas de Datos (Correlación Extrema):**",
            "La IA detectó que estas variables predecían al Target con más de un 85% de exactitud por sí solas. "
            "En MLOps, esto casi siempre es un 'Caballo de Troya' (ej. usar 'impuestos_pagados' para predecir si alguien es 'rico'). Destruyen la capacidad de generalizar."
        ),
        'nulos_masivos_eliminados': (
            "🕳️ **Degradación Masiva (>90% Nulos):**",
            "Se eliminaron porque carecen de señal estadística. Intentar imputar (rellenar) una variable donde falta el 90% "
            "de la información equivale a inventar datos, lo que induciría alucinaciones matemáticas en el modelo."
        ),
        'constantes_eliminadas': (
            "🧊 **Variables Constantes (Varianza Cero):**",
            "Se eliminaron porque tienen un único valor para casi todos los registros (ej. un dataset donde todos son del mismo país). "
            "Matemáticamente, si no hay variación, el algoritmo no puede trazar fronteras de decisión. Son peso muerto."
        ),
        'isomorficas_redundantes_eliminadas': (
            "👯 **Isomorfismo (Redundancia 1:1):**",
            "Se detectaron pares de columnas que dicen exactamente lo mismo en diferente formato (ej. 'ID_Ciudad' y 'Nombre_Ciudad'). "
            "Mantener ambas infla la dimensionalidad de la matriz, ralentiza el entrenamiento y causa multicolinealidad sin aportar nueva información."
        ),
        'ids_migrados_al_index': (
            "🔒 **Protección de Identidad (Migración al Index):**",
            "Los IDs no se eliminan, se protegen moviéndolos al índice de la matriz. Si se dejan como variables predictoras, "
            "los árboles de decisión (como Random Forest o LightGBM) 'memorizarán' a los pacientes por su ID en lugar de aprender los verdaderos patrones."
        )
    }

    # Renderizado Inteligente
    contenido_md = f"### 📜 Resolución Oficial: Justificación de Limpieza Estructural\n"
    contenido_md += f"**🎯 Target Protegido:** `{reporte.get('target_escogido', 'Desconocido')}`\n\n"

    if reporte.get('filas_sin_target_eliminadas', 0) > 0:
        contenido_md += f"> **Regla de Oro Aplicada:** Se aniquilaron **{reporte['filas_sin_target_eliminadas']} filas** porque su valor en el Target era nulo. Un modelo no puede aprender de una respuesta que no existe.\n\n"

    contenido_md += "---\n"

    operaciones_realizadas = 0

    # Recorremos el diccionario y solo mostramos las secciones donde la guillotina actuó
    for llave, (titulo, explicacion) in justificaciones.items():
        elementos = reporte.get(llave, [])
        if elementos:
            operaciones_realizadas += 1
            lista_formateada = ", ".join([f"`{e}`" for e in elementos])
            contenido_md += f"{titulo}\n"
            contenido_md += f"* **Decisión MLOps:** {explicacion}\n"
            contenido_md += f"* **Columnas Afectadas ({len(elementos)}):** {lista_formateada}\n\n"

    if operaciones_realizadas == 0:
        contenido_md += "✅ **Matriz Impecable:** La Guillotina evaluó la matriz y no encontró anomalías estructurales severas. Ninguna columna fue alterada.\n"

    # Mostrar en pantalla
    display(Markdown(contenido_md))


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'reporte_guillotina' not in locals() and 'reporte_guillotina' not in globals():
        raise EnvironmentError("No se encontró 'reporte_guillotina'. Ejecuta la Fase 1.3 primero.")

    # Ejecutamos el auditor
    auditar_dictamen_guillotina(reporte_guillotina)

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante: {env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Auditoría: {e}")


# In[30]:


# ==========================================
# 0. Blindaje de Dependencias UI y Matemáticas
# ==========================================
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import warnings

# ==========================================
# MOTOR MLOPS: AUTOPSIA DE FUGA DE DATOS (LEAKAGE)
# ==========================================
def autopsia_fuga_datos(df_original: pd.DataFrame, target_principal: str, columnas_sospechosas: list) -> None:
    """
    [FASE 1 - Paso 1.6] Autopsia de Fuga del Futuro (Target Leakage Proof).
    - MLOps Core: Demuestra estadísticamente por qué una variable es un "Caballo de Troya".
    - Conversión Inteligente: Factoriza variables categóricas automáticamente para medir correlación matemática.
    - Renderizado Táctico: Muestra cómo cambia el comportamiento de la variable sospechosa según la clase del Target.
    """
    if df_original is None or df_original.empty:
        print("🛑 Error Crítico: La matriz original está vacía o es inválida.")
        return

    if not columnas_sospechosas:
        print("✅ [BYPASS] No hay targets secundarios ni fugas detectadas para auditar.")
        return

    if target_principal not in df_original.columns:
        print(f"🛑 Error Crítico: El Target Principal '{target_principal}' no existe en la matriz.")
        return

    print("=== 🔮 FASE 1.6: Autopsia de Fuga del Futuro (Prueba de Fraude Predictivo) ===")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Preparación del Target (Lo pasamos a números si es texto para medir correlación)
        y_real = df_original[target_principal]
        if y_real.dtype == 'object' or y_real.dtype == 'category':
            y_numerico = pd.factorize(y_real)[0]
        else:
            y_numerico = y_real

        for col in columnas_sospechosas:
            if col not in df_original.columns:
                continue

            display(Markdown(f"### 🔍 Expediente de Fraude: `{col}` vs `{target_principal}`"))

            # 1. Blindaje contra Nulos para el cálculo matemático
            mask = ~df_original[col].isna() & ~y_real.isna()
            datos_limpios = df_original.loc[mask, col]
            y_limpio = y_numerico[mask]

            if len(datos_limpios) < 10:
                print(f"  ⚠️ Datos insuficientes para auditar '{col}'.")
                continue

            # 2. Factorización Inteligente si el rival también es categórico
            es_numerica = pd.api.types.is_numeric_dtype(datos_limpios)
            if not es_numerica:
                datos_numericos = pd.factorize(datos_limpios)[0]
            else:
                datos_numericos = datos_limpios

            # 3. Cálculo de Correlación (La prueba del delito)
            correlacion = np.abs(np.corrcoef(datos_numericos, y_limpio)[0, 1])

            # Veredicto de Correlación
            if correlacion >= 0.85:
                alerta = "🔴 FRAUDE EXTREMO (Clon del Target)"
            elif correlacion >= 0.50:
                alerta = "🟠 ALTO RIESGO (Bola de Cristal parcial)"
            else:
                alerta = "🟡 RIESGO ESTRUCTURAL (Variable redundante o Target secundario de negocio)"

            print(f"  📈 Correlación Matemática : {correlacion:.2%} -> {alerta}")
            print(f"  💡 Explicación MLOps    : Si la correlación es muy alta, el modelo simplemente memoriza esta columna y deja de pensar. Si no es alta, al ser un 'Target Secundario', representa un evento del futuro que no conocerás cuando llegue un cliente nuevo.\n")

            # 4. Tabla de Comportamiento Dinámico (¿Cómo delata al target?)
            print(f"  📊 Radiografía del Comportamiento (Promedios / Distribución por Clase de Target):")

            try:
                if es_numerica:
                    # Si el rival es numérico, agrupamos para ver cómo su promedio delata al target
                    resumen = df_original.groupby(target_principal)[col].agg(['mean', 'median', 'std']).reset_index()
                    resumen.columns = [f'Target ({target_principal})', 'Promedio', 'Mediana', 'Desviación']

                    tabla_estilizada = (
                        resumen.style
                        .format({'Promedio': '{:.2f}', 'Mediana': '{:.2f}', 'Desviación': '{:.2f}'})
                        .background_gradient(cmap='Oranges', subset=['Promedio'])
                        .hide(axis="index")
                    )
                else:
                    # Si el rival es categórico, hacemos una tabla cruzada (Crosstab)
                    resumen = pd.crosstab(df_original[target_principal], df_original[col], normalize='index') * 100

                    tabla_estilizada = (
                        resumen.style
                        .format("{:.1f}%")
                        .background_gradient(cmap='Oranges', axis=1)
                    )

                display(tabla_estilizada)
            except Exception as e:
                print(f"  ⚠️ No se pudo renderizar la tabla de cruce: {e}")

            print("-" * 80)


# ==========================================
# Celda de Ejecución Maestra en tu .ipynb
# ==========================================
try:
    # IMPORTANTE: Necesitamos 'df_sin_clones' (matriz ANTES de que la guillotina borrara todo)
    if 'df_sin_clones' not in locals() and 'df_sin_clones' not in globals():
        raise EnvironmentError("No se encontró 'df_sin_clones'. Ejecuta la Fase de Ingesta primero.")

    if 'reporte_guillotina' not in locals() and 'reporte_guillotina' not in globals():
        raise EnvironmentError("No se encontró 'reporte_guillotina'. Ejecuta la Fase 1.3 (Guillotina) primero.")

    # 1. Extraemos a los acusados (Targets secundarios + Fugas detectadas por la IA)
    target_rey = reporte_guillotina.get('target_escogido', '')

    acusados = []
    acusados.extend(reporte_guillotina.get('targets_secundarios_eliminados', []))
    acusados.extend(reporte_guillotina.get('fugas_datos_detectadas_ia', []))

    # 2. Ejecutamos el juicio visual
    if acusados and target_rey:
        print(f">>> ⚖️ LLEVANDO AL ESTRADO A {len(acusados)} VARIABLES ACUSADAS DE LEAKAGE <<<")
        autopsia_fuga_datos(
            df_original=df_sin_clones, 
            target_principal=target_rey,
            columnas_sospechosas=acusados
        )
    else:
        print("✅ No hay variables acusadas de ser fugas del futuro o targets secundarios para auditar.")

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante: {env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Autopsia de Fugas: {e}")


# In[31]:


# ==========================================
# 0. Blindaje de Dependencias UI y Matemáticas
# ==========================================
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import warnings

# ==========================================
# MOTOR MLOPS: AUTOPSIA FORENSE (PROFILING VIRTUAL)
# ==========================================
def autopsia_forense_variables(df_original: pd.DataFrame, columnas_condenadas: list) -> None:
    """
    [FASE 1 - Paso 1.5] Autopsia Forense de Variables (Análisis Post-Mortem).
    - Perfilado Matemático: Extrae la Moda, el Porcentaje absoluto y la frecuencia.
    - UI de Calor (Color Mapping): Aplica un gradiente térmico para resaltar visualmente el desbalance.
    - Escudo de Memoria (AutoML): Limita la tabla visual al Top 10 para evitar colapsar el Notebook.
    - Tolerancia a Nulos: Incluye los NaN en el cálculo de porcentajes para dar la imagen real.
    """
    if df_original is None or df_original.empty:
        print("🛑 Error Crítico: La matriz original está vacía o es inválida.")
        return

    if not columnas_condenadas:
        print("✅ [BYPASS] La lista de columnas a auditar está vacía. No hay autopsia necesaria.")
        return

    print("=== 🔬 FASE 1.5: Autopsia Forense de Variables (Radiografía de Varianza) ===")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for col in columnas_condenadas:
            if col not in df_original.columns:
                print(f"  ⚠️ Alerta: La columna '{col}' no existe en el DataFrame proporcionado.")
                continue

            # 1. Extracción de Datos Crudos
            serie = df_original[col]
            total_filas = len(serie)
            nulos = serie.isna().sum()

            # 2. Cálculos Estadísticos MLOps
            # Calculamos la Moda (El valor que más se repite)
            moda_serie = serie.mode(dropna=True)
            moda_val = moda_serie.iloc[0] if not moda_serie.empty else "N/A (100% Nulo)"

            # Distribución absoluta y relativa (Top 10 para blindaje de RAM UI)
            conteo = serie.value_counts(dropna=False).head(10)
            porcentajes = serie.value_counts(dropna=False, normalize=True).head(10)

            # 3. Construcción de la Tabla de Autopsia
            df_reporte = pd.DataFrame({
                'Valor / Categoría': conteo.index.astype(str),
                'Frecuencia (Filas)': conteo.values,
                'Porcentaje (%)': porcentajes.values * 100
            })

            # 4. Inteligencia de UI: Estilizado Térmico (Color Gradient)
            # Entre más alto el porcentaje, más oscuro será el color (usamos la paleta Reds/Rojos)
            tabla_estilizada = (
                df_reporte.style
                .format({
                    'Frecuencia (Filas)': '{:,.0f}', 
                    'Porcentaje (%)': '{:.3f}%'
                })
                .background_gradient(cmap='Reds', subset=['Porcentaje (%)'])
                .set_caption(f"Distribución Top 10 de la variable '{col}'")
                .hide(axis="index") # Ocultamos el index por defecto para mayor limpieza visual
            )

            # 5. Renderizado Ejecutivo
            display(Markdown(f"### 🩻 Análisis Post-Mortem: `{col}`"))
            print(f"  📌 Valor Dominante (Moda) : {moda_val}")
            print(f"  🕳️ Total de Datos Nulos   : {nulos:,} ({nulos/total_filas:.2%})")
            print(f"  🔍 Renderizando distribución...\n")

            display(tabla_estilizada)
            print("-" * 80)


# ==========================================
# Celda de Ejecución Maestra en tu .ipynb
# ==========================================
try:
    # IMPORTANTE: Le pasamos 'df_sin_clones' (la matriz ANTES de que la guillotina borrara todo)
    if 'df_sin_clones' not in locals() and 'df_sin_clones' not in globals():
        raise EnvironmentError("No se encontró 'df_sin_clones'. Este método necesita la matriz original para leer la columna borrada.")

    if 'reporte_guillotina' not in locals() and 'reporte_guillotina' not in globals():
        raise EnvironmentError("No se encontró 'reporte_guillotina'. Ejecuta la Fase 1.3 primero.")

    # Extraemos automáticamente las variables constantes que la guillotina condenó
    # (Por ejemplo: 'entrada_es_home')
    columnas_constantes = reporte_guillotina.get('constantes_eliminadas', [])

    if columnas_constantes:
        print(f">>> 🩺 INICIANDO AUTOPSIA PARA {len(columnas_constantes)} VARIABLES CONSTANTES <<<")
        autopsia_forense_variables(
            df_original=df_sin_clones, 
            columnas_condenadas=columnas_constantes
        )
    else:
        print("✅ No se detectaron variables constantes para auditar en el reporte previo.")

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante: {env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Autopsia Forense: {e}")


# In[32]:


df_purgado.info()


# In[33]:


df_purgado.head()


# In[34]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import numpy as np
import re
import time
import warnings # 🔧 NUEVO: Módulo para controlar las alertas de la consola
from typing import Tuple

def downcasting_matematico_inteligente(
    df: pd.DataFrame, 
    umbral_categoria: float = 0.50,
    target_col: str = None
) -> Tuple[pd.DataFrame, dict]:
    """
    [FASE 1 - Paso 1.4] Downcasting Matemático (Nivel AutoML).
    - Regla Target: Convierte el target directamente a 'category' blindándolo.
    - Detección Temporal Avanzada con Auto-Limpieza (NUEVO): Elimina basura léxica ('?', '*', etc.) de fechas y horas automáticamente.
    - Auto-Parsing IoT: Detecta columnas numéricas que en realidad son fechas ocultas.
    - Detección Booleana Oculta: Usa Regex para filtrar basura y mapea textos 'yes/no' a 'boolean'.
    - Compresión Numérica: Reduce float64 a float32 y asigna tipos INT firmados (int8, 16, 32).
    - Muro Anti-Objetos: Obliga a todo texto sobreviviente a ser 'category' para proteger LightGBM.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df, {}

    print("=== 🗜️ FASE 1.4: Downcasting Matemático Inteligente (IA de Contenido) ===")

    # 🚀 FIX MLOps: SILENCIADOR BLINDADO CONTRA AVISOS DE FECHAS Y REGEX
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Could not infer format.*")

    inicio_timer = time.time()
    df_opt = df.copy()

    mem_antes = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    filas_totales = len(df_opt)
    contadores = {'int': 0, 'float': 0, 'category': 0, 'datetime': 0, 'bool': 0}

    print(" 🔍 Escaneando contenido para inferencia de tipos profundos...")

    # Motor Regex para Falsos Nulos
    patron_nulos_ocultos = re.compile(r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|)$|^[^a-zA-Z0-9]+$')

    for col in df_opt.columns:
        tipo_actual = df_opt[col].dtype
        unicos_count = df_opt[col].nunique(dropna=False)

        # ==========================================
        # 1. EL TARGET ES REY (Blindaje Prioritario)
        # ==========================================
        if target_col and col == target_col:
            if df_opt[col].dtype.name != 'category':
                df_opt[col] = df_opt[col].astype('category')
                contadores['category'] += 1
                print(f"    🎯 Target '{col}' blindado y convertido a categoría.")
            continue

        # ==========================================
        # 2. Inferencia Profunda en Textos (Objects)
        # ==========================================
        if pd.api.types.is_object_dtype(tipo_actual) or pd.api.types.is_string_dtype(tipo_actual):

            valores_limpios = df_opt[col].dropna()
            if valores_limpios.empty: continue

            valores_puros = valores_limpios.astype(str).str.lower().str.strip()
            unicos_texto = set(valores_puros.unique())

            # Escudo Regex contra Falsos Nulos
            textos_reales = {x for x in unicos_texto if not patron_nulos_ocultos.match(x)}

            # A. Detección de Booleanos Ocultos en Texto
            diccionario_bool = {
                'yes': True, 'no': False, 
                'si': True, 'true': True, 'false': False, 'verdadero': True, 'falso': False,
                't': True, 'f': False, 'y': True, 'n': False,
                '1': True, '0': False
            }

            if textos_reales and textos_reales.issubset(diccionario_bool.keys()):
                df_opt[col] = df_opt[col].astype(str).str.lower().str.strip().map(diccionario_bool)
                df_opt[col] = df_opt[col].astype('boolean') 
                contadores['bool'] += 1
                print(f"    ⚖️ Texto Booleano detectado en '{col}': Convertido a boolean (Soporta Nulos).")
                continue

            # 🚀 B. Detección Heurística de Fechas y Tiempos (Super-Regex + Auto-Limpieza)
            muestra = valores_puros.head(50)

            # 🛡️ ESCUDO DE AUTO-LIMPIEZA: Simulamos quitar caracteres extraños de la muestra
            # Mantenemos números, letras (AM/PM, Meses), espacios y separadores típicos de tiempo (- / : .)
            muestra_limpia = muestra.str.replace(r'[^0-9a-zA-Z\s\-\/:\.]', '', regex=True).str.strip()

            patron_fecha_clasica = r'(?i)[-/:]|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
            patron_unix_txt = r'^1\d{9}(?:\.\d+)?$|^1\d{12}(?:\.\d+)?$'

            es_fecha_clasica = muestra_limpia.str.contains(patron_fecha_clasica, regex=True).any()
            es_unix_txt = muestra_limpia.str.contains(patron_unix_txt, regex=True).any()

            if len(muestra_limpia) > 0 and (es_fecha_clasica or es_unix_txt):
                try:
                    # Probamos la conversión matemática en el entorno seguro (muestra)
                    if es_unix_txt:
                        prueba_fecha = pd.to_datetime(pd.to_numeric(muestra_limpia, errors='coerce'), unit='s', errors='coerce')
                    else:
                        prueba_fecha = pd.to_datetime(muestra_limpia, errors='coerce', dayfirst=True)

                    if prueba_fecha.notna().mean() >= 0.80:
                        # 🚀 ¡Aprobado! Aplicamos la limpieza léxica a toda la columna real (salvaguardando los NaNs)
                        mask_viva = df_opt[col].notna()
                        df_opt.loc[mask_viva, col] = (
                            df_opt.loc[mask_viva, col]
                            .astype(str)
                            .str.replace(r'[^0-9a-zA-Z\s\-\/:\.]', '', regex=True)
                            .str.strip()
                        )

                        # Conversión Final
                        if es_unix_txt:
                            df_opt[col] = pd.to_datetime(pd.to_numeric(df_opt[col], errors='coerce'), unit='s', errors='coerce')
                            print(f"    🕒 Unix Timestamp Textual detectado en '{col}': Auto-limpiado y convertido a Datetime.")
                        else:
                            df_opt[col] = pd.to_datetime(df_opt[col], errors='coerce', dayfirst=True)
                            print(f"    📅 Fecha/Hora detectada en '{col}': Basura purgada y convertida a Datetime (NaNs protegidos).")
                        contadores['datetime'] += 1
                        continue
                except Exception:
                    pass 

            # C. FIX MLOPS: Optimización y Forzado de Categóricas
            ratio_unicidad = unicos_count / filas_totales
            if ratio_unicidad < umbral_categoria:
                df_opt[col] = df_opt[col].astype('category')
                contadores['category'] += 1
            else:
                # El Muro Anti-Objetos: Forzamos la conversión para proteger Fases futuras
                df_opt[col] = df_opt[col].astype('category')
                contadores['category'] += 1
                print(f"    ⚠️ '{col}' tiene alta cardinalidad ({ratio_unicidad:.1%}). Forzado a 'category' por seguridad algorítmica.")

        # ==========================================
        # 3. Compresión de Numéricas y Booleanas Nativas
        # ==========================================
        elif pd.api.types.is_numeric_dtype(tipo_actual):
            # 🚀 3.0 Detección Heurística de Unix Timestamps Numéricos (Sensores IoT)
            muestra_num = df_opt[col].dropna().head(50)
            if len(muestra_num) > 0:
                es_unix_segundos = (muestra_num >= 631152000).all() and (muestra_num <= 2208988800).all()
                es_unix_milisegundos = (muestra_num >= 631152000000).all() and (muestra_num <= 2208988800000).all()

                if (es_unix_segundos or es_unix_milisegundos) and unicos_count > 2:
                    unidad_tiempo = 'ms' if es_unix_milisegundos else 's'
                    df_opt[col] = pd.to_datetime(df_opt[col], unit=unidad_tiempo, errors='coerce')
                    contadores['datetime'] += 1
                    print(f"    🕒 Unix Timestamp Numérico ({unidad_tiempo}) detectado en '{col}': Convertido a Datetime IoT.")
                    continue

            c_min = df_opt[col].min()
            c_max = df_opt[col].max()
            tiene_nulos = df_opt[col].isna().any()

            # 3.1 Detección Booleana Numérica Nativa
            unicos_numericos = df_opt[col].dropna().unique()
            if set(unicos_numericos).issubset({0, 1, 0.0, 1.0}):
                df_opt[col] = df_opt[col].astype('boolean') 
                contadores['bool'] += 1
                continue

            # 3.2 Compresión de Enteros (Signed INT exactos)
            if pd.api.types.is_integer_dtype(tipo_actual) and not tiene_nulos:
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df_opt[col] = df_opt[col].astype(np.int32)

                if df_opt[col].dtype != tipo_actual:
                    contadores['int'] += 1

            # 3.3 Compresión de Flotantes 
            elif pd.api.types.is_float_dtype(tipo_actual):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df_opt[col] = df_opt[col].astype(np.float32)
                    contadores['float'] += 1

    # ==========================================
    # D. Reporte Ejecutivo de MLOps
    # ==========================================
    mem_despues = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    ahorro_mb = mem_antes - mem_despues
    porcentaje_ahorro = 100 * (ahorro_mb / mem_antes) if mem_antes > 0 else 0

    # 🔧 Limpieza final: Restauramos las advertencias generales para el resto del cuaderno
    warnings.filterwarnings("default", category=UserWarning)

    print(f"\n✅ Compresión Matemática Completada en {time.time() - inicio_timer:.3f}s:")
    print(f"  📉 Transformaciones: {contadores['int']} Int | {contadores['float']} Float | {contadores['category']} Cat | {contadores['datetime']} Date | {contadores['bool']} Bool")
    print(f"  💾 Memoria Inicial : {mem_antes:.2f} MB")
    print(f"  💽 Memoria Final   : {mem_despues:.2f} MB (-{porcentaje_ahorro:.1f}%)")

    metricas_ram = {
        'mem_inicial_mb': mem_antes,
        'mem_final_mb': mem_despues,
        'ahorro_mb': ahorro_mb,
        'porcentaje_ahorro': porcentaje_ahorro
    }

    return df_opt, metricas_ram

# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Ejecuta La Guillotina (Paso 1.3) primero.")
    if 'reporte_guillotina' not in locals() and 'reporte_guillotina' not in globals():
        raise EnvironmentError("No se encontró 'reporte_guillotina'. Ejecuta La Guillotina (Paso 1.3) primero.")

    # 🔗 CONEXIÓN MLOps: Extraemos el target dinámicamente del paso anterior
    target_heredado = reporte_guillotina.get('target_escogido')

    if target_heredado:
        print(f">>> 🔗 Conectando MLOps: Heredando Target '{target_heredado}' desde la Guillotina <<<")
    else:
        print(">>> ⚠️ Advertencia: No se heredó ningún Target de la Guillotina. Procesando sin escudo. <<<")

    df_purgado, metricas_downcast = downcasting_matematico_inteligente(
        df=df_purgado,
        umbral_categoria=0.50,
        target_col=target_heredado # <- Alimentación dinámica
    )

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en el Downcasting Matemático: {e}")


# In[35]:


df_purgado.info()


# In[36]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import re
import time

def auditar_y_sanitizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    [FASE 2 - Paso 2.1] Detección de Tipos y Prevención de Crash.
    - Sanitización Regex: Limpia espacios, tildes y caracteres especiales de los nombres de columnas.
    - Resolución de Duplicados: Detecta nombres idénticos y les asigna un sufijo (evita el colapso de LightGBM/FLAML).
    - Auditoría de Tipos: Genera un reporte rápido de la integridad del esquema.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df

    print("=== 🛡️ FASE 2.1: Auditoría de Integridad y Sanitización de Columnas ===")

    inicio_timer = time.time()
    df_opt = df.copy()

    # ==========================================
    # 1. Motor Regex de Sanitización (Formato Producción)
    # ==========================================
    # Convierte "Edad del Cliente (%)" a "edad_del_cliente"
    nombres_originales = list(df_opt.columns)
    nombres_limpios = []

    for col in nombres_originales:
        # Convertimos a string, minúsculas y quitamos espacios a los lados
        col_str = str(col).strip().lower()
        # Reemplazamos cualquier cosa que NO sea letra o número por un guion bajo
        col_str = re.sub(r'[^a-z0-9_]', '_', col_str)
        # Eliminamos guiones bajos múltiples consecutivos (ej. '__' a '_')
        col_str = re.sub(r'_+', '_', col_str)
        # Quitamos guiones bajos al principio o al final
        col_str = col_str.strip('_')
        nombres_limpios.append(col_str)

    df_opt.columns = nombres_limpios

    # ==========================================
    # 2. Escudo Anti-Crash (Resolución de Duplicados)
    # ==========================================
    columnas_finales = []
    dicc_vistos = {}
    duplicados_corregidos = 0

    for col in df_opt.columns:
        if col not in dicc_vistos:
            dicc_vistos[col] = 0
            columnas_finales.append(col)
        else:
            dicc_vistos[col] += 1
            duplicados_corregidos += 1
            # Si "ingreso" ya existe, lo llama "ingreso_v1"
            nuevo_nombre = f"{col}_v{dicc_vistos[col]}"
            columnas_finales.append(nuevo_nombre)
            print(f"  ⚠️ Peligro de Crash Evitado: Columna '{col}' renombrada a '{nuevo_nombre}'")

    df_opt.columns = columnas_finales

    # ==========================================
    # 3. Auditoría de Tipos (Alternativa Limpia a .info)
    # ==========================================
    conteo_tipos = df_opt.dtypes.astype(str).value_counts().to_dict()

    tiempo_total = time.time() - inicio_timer
    print(f"\n✅ Auditoría Estructural Completada en {tiempo_total:.3f}s:")
    print(f"  ✨ Columnas Sanitizadas : {len(df_opt.columns)}")
    if duplicados_corregidos > 0:
        print(f"  🩹 Duplicados Resueltos : {duplicados_corregidos} (Prevención LightGBM activada)")
    else:
        print(f"  ✔️ Duplicados           : Ninguno detectado (Esquema saludable)")

    print("\n  📊 Mapa de Tipos de Datos (dtypes):")
    for tipo, cantidad in conteo_tipos.items():
        print(f"     - {tipo.ljust(12)}: {cantidad} columnas")

    return df_opt


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    # 💡 Lógica de Cadena de Montaje: Recibimos de df_purgado y lo MANTENEMOS como df_purgado
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Ejecuta el paso 1.4 primero.")

    # Ejecutamos la sanitización y sobrescribimos la matriz
    df_purgado = auditar_y_sanitizar_columnas(df=df_purgado)

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Auditoría de Columnas: {e}")


# In[37]:


df_purgado[0:9]


# In[38]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import time
from typing import Tuple

def normalizar_categoricas_y_fusionar(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    [FASE 2 - Paso 2.2] Normalización de Categóricas y Corrección por Mayoría.
    - Limpieza Textual: Minúsculas, sin tildes, strip.
    - Escáner de Espacios Internos: Fusiona tokens como "<=50 k" a "<=50k".
    - Fusión Canónica: Al mapear el texto sucio a su versión limpia, los errores 
      tipográficos se agrupan automáticamente, sumando sus frecuencias y 
      preservando la distribución estadística real.
    - Zero-RAM Overhead: Trabaja sobre los diccionarios de categorías, no sobre la matriz entera.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df, {}

    print("=== 🧹 FASE 2.2: Normalización Categórica y Fusión Tipográfica ===")

    inicio_timer = time.time()
    df_opt = df.copy()
    contadores = {'procesadas': 0, 'conflictos_resueltos': 0}

    for col in df_opt.columns:
        tipo = df_opt[col].dtype

        # Filtro: Solo actuamos si la variable es de texto o una categoría de Pandas
        if pd.api.types.is_object_dtype(tipo) or pd.api.types.is_string_dtype(tipo) or isinstance(tipo, pd.CategoricalDtype):

            # 1. Extracción Eficiente (Extraemos solo los valores únicos para no saturar RAM)
            if isinstance(tipo, pd.CategoricalDtype):
                categorias_crudas = df_opt[col].cat.categories
            else:
                categorias_crudas = df_opt[col].dropna().unique()

            if len(categorias_crudas) == 0:
                continue

            # Convertimos a Series para usar la API vectorizada de Pandas .str
            s_limpia = pd.Series(categorias_crudas).astype(str)

            # ==========================================
            # 2. MOTOR REGEX DE SANITIZACIÓN MULTICAPA
            # ==========================================
            # A. Minúsculas
            s_limpia = s_limpia.str.lower()

            # B. Remover Tildes y Acentos (Normalización NFKD)
            s_limpia = s_limpia.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

            # C. Colapsar múltiples espacios seguidos a uno solo
            s_limpia = s_limpia.str.replace(r'\s+', ' ', regex=True)

            # D. Remover espacios alrededor de símbolos (Ej: "<= 50" -> "<=50")
            s_limpia = s_limpia.str.replace(r'\s*([^\w\s])\s*', r'\1', regex=True)

            # E. Remover espacios entre números y letras (Ej: "50 k" -> "50k")
            s_limpia = s_limpia.str.replace(r'(?<=\d)\s+(?=[a-z])|(?<=[a-z])\s+(?=\d)', '', regex=True)

            # F. Strip final y convertir espacios restantes a guiones bajos (Ej: "united states" -> "united_states")
            s_limpia = s_limpia.str.strip().str.replace(r'\s+', '_', regex=True)

            # ==========================================
            # 3. FUSIÓN Y MAPEO (Corrección por Mayoría)
            # ==========================================
            # Creamos un diccionario: { " <=50 k " : "<=50k", "<=50K" : "<=50k" }
            mapper = dict(zip(categorias_crudas, s_limpia))

            # Calculamos cuántas clases "basura" se agruparon en una clase canónica
            unicos_antes = len(categorias_crudas)
            unicos_despues = s_limpia.nunique()
            conflictos = unicos_antes - unicos_despues

            # Aplicamos el diccionario directamente a la matriz
            # Al usar .map(), los NaNs originales se respetan y se quedan como NaNs.
            if isinstance(tipo, pd.CategoricalDtype):
                df_opt[col] = df_opt[col].map(mapper).astype('category')
            else:
                df_opt[col] = df_opt[col].map(mapper)

            contadores['procesadas'] += 1
            contadores['conflictos_resueltos'] += conflictos

            if conflictos > 0:
                print(f"    🧬 '{col}': {conflictos} variantes tipográficas fusionadas.")

    tiempo_total = time.time() - inicio_timer

    print(f"\n✅ Normalización Textual Completada en {tiempo_total:.3f}s:")
    print(f"  📊 Columnas Procesadas  : {contadores['procesadas']}")
    print(f"  🩹 Conflictos Resueltos : {contadores['conflictos_resueltos']} (Clases canónicas consolidadas)")

    return df_opt, contadores


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    # 💡 Lógica de Cadena de Montaje Estricta
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Asegúrate de ejecutar el Paso 2.1 primero.")

    # SOBRESCRIBIMOS df_purgado para heredar las categorías limpias
    df_purgado, reporte_cat = normalizar_categoricas_y_fusionar(df=df_purgado)

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Normalización de Categóricas: {e}")


# In[39]:


df_purgado[0:9]


# In[40]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple

def rescate_numerico_y_compresion(
    df: pd.DataFrame, 
    target_principal: str, # 👑 NUEVO: El nombre del Target para blindarlo
    tolerancia_destruccion: float = 0.05
) -> Tuple[pd.DataFrame, dict]:
    """
    [FASE 2 - Paso 2.3] Casteo Forzado y Rescate de Falsos Textos.
    - Escudo del Rey: Inmunidad absoluta para la variable Target. Jamás será casteada.
    - Motor Regex: Limpia símbolos de moneda, porcentajes y comas miliares.
    - Casteo Coerce: Fuerza la conversión a numérico aislando textos irreconocibles como NaNs.
    - Rollback AutoML: Si la conversión genera demasiados NaNs (>5%), revierte los cambios.
    - Downcasting Integrado: Al rescatar el número, evalúa min/max y asigna el float ideal.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("🛑 Error Crítico: La matriz está vacía o es inválida.")
        return df, {}

    if target_principal not in df.columns:
        print(f"🛑 Error Crítico: El Target '{target_principal}' no se encontró en la matriz.")
        return df, {}

    print("=== 🚑 FASE 2.3: Rescate de Falsos Textos y Compresión Flotante ===")

    inicio_timer = time.time()
    df_opt = df.copy()
    filas_totales = len(df_opt)

    mem_antes = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    contadores = {'rescatadas_float32': 0, 'rescatadas_float64': 0, 'ignoradas_texto_real': 0, 'target_protegido': 1}

    print(" 🔍 Buscando números disfrazados de texto...")

    for col in df_opt.columns:
        # 👑 ESCUDO DEL REY: Si es el Target, lo saltamos inmediatamente
        if col == target_principal:
            print(f"   🛡️ [INMUNIDAD] Columna Target '{col}' protegida. Omitiendo casteo.")
            continue

        tipo_actual = df_opt[col].dtype

        # Solo intentamos el rescate en columnas que son texto o categorías
        if pd.api.types.is_object_dtype(tipo_actual) or pd.api.types.is_string_dtype(tipo_actual) or isinstance(tipo_actual, pd.CategoricalDtype):

            # 1. Snapshot de Seguridad (Rollback)
            nulos_originales = df_opt[col].isna().sum()

            # 2. Extracción y Limpieza Regex
            # Reemplazamos símbolos comunes que disfrazan números ($, €, £, %, comas miliares y espacios)
            serie_limpia = df_opt[col].astype(str).str.replace(r'[$,€£%\s]', '', regex=True)

            # 3. Casteo Forzado
            serie_numerica = pd.to_numeric(serie_limpia, errors='coerce')

            # 4. Auditoría de Destrucción (¿Era realmente un número?)
            nulos_nuevos = serie_numerica.isna().sum()
            tasa_destruccion = (nulos_nuevos - nulos_originales) / filas_totales

            # Si se destruyó menos del 5% de los datos, ¡era un falso texto! Procedemos.
            if tasa_destruccion <= tolerancia_destruccion:

                c_min = serie_numerica.min()
                c_max = serie_numerica.max()

                # 5. Downcasting Integrado Inteligente (Asignación del Float Ideal)
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df_opt[col] = serie_numerica.astype(np.float32)
                    contadores['rescatadas_float32'] += 1
                else:
                    df_opt[col] = serie_numerica.astype(np.float64)
                    contadores['rescatadas_float64'] += 1

                print(f"    🚑 Rescate Exitoso: '{col}' convertida a {df_opt[col].dtype}")

            else:
                # Era una categoría de texto real (ej. "Married-civ-spouse"). Abortamos y protegemos.
                contadores['ignoradas_texto_real'] += 1

    # ==========================================
    # Reporte Ejecutivo de MLOps
    # ==========================================
    mem_despues = df_opt.memory_usage(deep=True).sum() / (1024 ** 2)
    ahorro_mb = mem_antes - mem_despues
    porcentaje_ahorro = 100 * (ahorro_mb / mem_antes) if mem_antes > 0 else 0

    tiempo_total = time.time() - inicio_timer
    total_rescatadas = contadores['rescatadas_float32'] + contadores['rescatadas_float64']

    print(f"\n✅ Rescate y Casteo Completado en {tiempo_total:.3f}s:")
    if total_rescatadas > 0:
        print(f"  📉 Columnas Rescatadas : {total_rescatadas} ({contadores['rescatadas_float32']} float32 | {contadores['rescatadas_float64']} float64)")
    else:
        print(f"  ✔️ Columnas Rescatadas : 0 (No se detectaron falsos textos)")

    print(f"  🛡️ Protecciones Activas: {contadores['ignoradas_texto_real']} textos reales ignorados con éxito")
    print(f"  👑 Target Protegido    : Sí ('{target_principal}')")
    print(f"  💾 Memoria Inicial     : {mem_antes:.2f} MB")
    print(f"  💽 Memoria Final       : {mem_despues:.2f} MB (-{porcentaje_ahorro:.1f}%)")

    return df_opt, contadores


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Ejecuta el paso anterior primero.")

    if 'target_ganador_fase1' not in locals() and 'target_ganador_fase1' not in globals():
        raise EnvironmentError("No se encontró la variable 'target_ganador_fase1' definida en el Paso 1.3.")

    # SOBRESCRIBIMOS df_purgado para mantener la linealidad del pipeline
    df_purgado, reporte_rescate = rescate_numerico_y_compresion(
        df=df_purgado,
        target_principal=target_ganador_fase1, # 👑 Pasamos dinámicamente el ganador de la guillotina
        tolerancia_destruccion=0.05 # Si falla en >5% de filas, asume que es texto puro y no lo toca
    )

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en el Rescate Numérico: {e}")


# In[41]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import time
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

def validacion_adversaria_automl(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    target_col: str,
    umbral_auc: float = 0.60
) -> Tuple[List[str], float]:
    """
    [FASE 3 - Paso 3.1] Validación Adversaria (Detector de Concept Drift).
    - Objetivo: Entrenar un LightGBM para distinguir entre Train y Test.
    - Inteligencia: Si el AUC > umbral, extrae las variables culpables del drift.
    - Blindaje: Ignora automáticamente la variable objetivo real para no hacer trampa.
    - Pre-procesamiento: Descompone variables Datetime en numéricas para evitar crasheos de LightGBM.
    - MLOps: Retorna la lista de variables tóxicas para ejecutarlas en la guillotina.
    """
    if df_train.empty or df_test.empty:
        print("🛑 Error Crítico: Uno de los DataFrames está vacío.")
        return [], 0.0

    print("=== 🕵️‍♂️ FASE 3.1: Validación Adversaria (Train vs Test) ===")

    inicio_timer = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ==========================================
        # 1. Preparación del Escenario Adversario
        # ==========================================
        # Copiamos para no alterar los originales
        X_tr = df_train.copy()
        X_te = df_test.copy()

        # Eliminamos el Target real de negocio (ej. 'income') si existe, 
        # porque el Test no lo debería tener (o no debemos usarlo aquí)
        if target_col in X_tr.columns: X_tr.drop(columns=[target_col], inplace=True)
        if target_col in X_te.columns: X_te.drop(columns=[target_col], inplace=True)

        # Alineamos columnas por si el Test viene con menos variables
        columnas_comunes = list(set(X_tr.columns).intersection(set(X_te.columns)))
        X_tr = X_tr[columnas_comunes]
        X_te = X_te[columnas_comunes]

        # Creamos el Target Adversario: 0 = Train, 1 = Test
        X_tr['is_test'] = 0
        X_te['is_test'] = 1

        # Unimos todo en un solo DataFrame
        df_adversario = pd.concat([X_tr, X_te], axis=0, ignore_index=True)

        # 🚀 NUEVO ESCUDO: Descomposición de Datetimes para LightGBM
        cols_datetime = df_adversario.select_dtypes(include=['datetime64', 'datetimetz']).columns
        if len(cols_datetime) > 0:
            print(f"    🗓️ Descomponiendo {len(cols_datetime)} variables Datetime para LightGBM...")
            for col in cols_datetime:
                df_adversario[f'{col}_year'] = df_adversario[col].dt.year
                df_adversario[f'{col}_month'] = df_adversario[col].dt.month
                df_adversario[f'{col}_day'] = df_adversario[col].dt.day
                df_adversario[f'{col}_dayofweek'] = df_adversario[col].dt.dayofweek
            # Destruimos las originales que causan el crasheo
            df_adversario.drop(columns=cols_datetime, inplace=True)

        # 🚀 NUEVO ESCUDO 2: Compatibilidad de Booleanos
        # LightGBM prefiere los booleanos nativos de pandas como numéricos o categóricos
        cols_bool = df_adversario.select_dtypes(include=['boolean']).columns
        if len(cols_bool) > 0:
            for col in cols_bool:
                df_adversario[col] = df_adversario[col].astype('float32') # Float soporta NaNs y LightGBM lo entiende

        y_adv = df_adversario['is_test']
        X_adv = df_adversario.drop(columns=['is_test'])

        # ==========================================
        # 2. División Interna para Evaluación Justa
        # ==========================================
        # Separamos 30% solo para medir el AUC del modelo adversario
        X_adv_train, X_adv_val, y_adv_train, y_adv_val = train_test_split(
            X_adv, y_adv, test_size=0.30, random_state=42, stratify=y_adv
        )

        # ==========================================
        # 3. Entrenamiento del Modelo Espía (LightGBM)
        # ==========================================
        print("  🚀 Entrenando LightGBM Adversario...")

        # LightGBM es ideal porque detecta las variables 'category' nativamente
        modelo_adv = lgb.LGBMClassifier(
            n_estimators=50,      # Rápido, solo queremos ver si hay un patrón obvio
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            verbose=-1            # Muteamos los warnings de C++
        )

        modelo_adv.fit(X_adv_train, y_adv_train)

        # ==========================================
        # 4. Veredicto del Tribunal (AUC y SHAP/Gain)
        # ==========================================
        preds = modelo_adv.predict_proba(X_adv_val)[:, 1]
        auc_score = roc_auc_score(y_adv_val, preds)

        variables_a_neutralizar = []

        print(f"  ⚖️ Veredicto del AUC: {auc_score:.4f} (Umbral de peligro: {umbral_auc})")

        if auc_score > umbral_auc:
            print("  🚨 PELIGRO: Concept Drift Detectado. El modelo puede distinguir Train de Test.")

            # Extraemos la importancia de las variables (Feature Importance by Gain)
            importancias = pd.DataFrame({
                'Variable': X_adv.columns,
                'Importancia': modelo_adv.feature_importances_
            }).sort_values(by='Importancia', ascending=False)

            # Regla AutoML: Tomamos las variables que acumulan el 80% de la importancia del drift
            importancias['Acumulado'] = importancias['Importancia'].cumsum() / importancias['Importancia'].sum()
            variables_toxicas = importancias[importancias['Acumulado'] <= 0.80]['Variable'].tolist()

            # Si solo una variable causa el 100% del drift, la lista podría estar vacía, la forzamos
            if not variables_toxicas:
                variables_toxicas = [importancias.iloc[0]['Variable']]

            variables_a_neutralizar = variables_toxicas
            print(f"  🪓 Variables Tóxicas marcadas para la guillotina: {variables_a_neutralizar}")

        else:
            print("  ✔️ Matriz Segura. Train y Test provienen de la misma distribución estadística.")

    tiempo_total = time.time() - inicio_timer
    print(f"  ⏱️ Tiempo de validación: {tiempo_total:.2f}s")

    # Mapeo Inverso: Si una variable tóxica fue 'Date_year', significa que la original 'Date' debe morir
    variables_finales_a_neutralizar = set()
    for var_tox in variables_a_neutralizar:
        if '_year' in var_tox or '_month' in var_tox or '_day' in var_tox or '_dayofweek' in var_tox:
            base_col = var_tox.rsplit('_', 1)[0]
            if base_col in df_train.columns:
                variables_finales_a_neutralizar.add(base_col)
        else:
            variables_finales_a_neutralizar.add(var_tox)

    return list(variables_finales_a_neutralizar), auc_score


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Ejecuta La Guillotina (Paso 1.3) primero.")

    # ==========================================
    # 🧠 AUTO-DETECCIÓN DEL TARGET (MLOps Wiring)
    # ==========================================
    # El código extrae la variable objetivo del diccionario que devolvió el Paso 1.3
    if 'reporte_guillotina' in locals() and reporte_guillotina.get('target_escogido'):
        target_detectado = reporte_guillotina['target_escogido']
    elif 'mis_targets' in locals() and len(mis_targets) > 0:
        target_detectado = mis_targets[0] 
    else:
        raise ValueError("🛑 No se encontró el Target ('reporte_guillotina' o 'mis_targets') de las fases previas.")

    print(f"🎯 Target auto-detectado del pipeline: '{target_detectado}'")

    # 💡 PARADOJA DE UN SOLO DATASET:
    print("--- 🔬 Simulando un entorno Train/Test para probar la validación ---")
    df_train_simulado, df_test_simulado = train_test_split(df_purgado, test_size=0.20, random_state=99)

    vars_toxicas, score_auc = validacion_adversaria_automl(
        df_train=df_train_simulado,
        df_test=df_test_simulado,
        target_col=target_detectado, # <--- Se inyecta automáticamente aquí
        umbral_auc=0.60
    )

    # Si detectamos variables del futuro/drift, las aniquilamos del dataset principal de una vez
    if vars_toxicas:
        print(f"\n  🔪 Ejecutando Neutralización en df_purgado...")
        df_purgado.drop(columns=vars_toxicas, inplace=True, errors='ignore')
        print(f"  ✅ Variables {vars_toxicas} eliminadas de la matriz principal.")

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en la Validación Adversaria: {e}")


# In[42]:


df_purgado.head()


# In[43]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import time
import re # 🚀 NUEVO: Importamos regex para la limpieza profunda
from typing import Tuple, Dict, List

def aislar_target_y_enrutar(
    df: pd.DataFrame, 
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    [FASE 3 - Paso 3.2] Aislamiento del Target y Regla de Oro.
    - Auto-Corrección Extendida: Detecta el target ignorando mayúsculas, espacios, guiones y guiones bajos.
    - Regla Estricta: Purga (elimina) cualquier registro donde la variable objetivo sea nula.
    - Aislamiento Temprano: Separa la matriz en características (X) y objetivo (y).
    - Enrutamiento (AutoML): Escanea los dtypes y crea listas explícitas de variables.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("🛑 Error Crítico: La matriz está vacía o es inválida.")

    # ==========================================
    # 🚀 NUEVO: Auto-Corrección Inteligente de Columnas (Bulletproof)
    # ==========================================
    if target_col not in df.columns:
        # Función destructiva: borra todo lo que no sea letra o número para una comparación pura
        def normalizar(nombre):
            return re.sub(r'[^a-z0-9]', '', str(nombre).lower())

        target_norm = normalizar(target_col) # 'No-show' se convierte en 'noshow'
        mapa_cols = {normalizar(c): c for c in df.columns}

        if target_norm in mapa_cols:
            target_real = mapa_cols[target_norm]
            print(f"  🪄 [AUTO-CORRECCIÓN] Target original '{target_col}' mapeado a -> '{target_real}'")
            target_col = target_real # Actualizamos la variable para usar la que sí existe en el DataFrame
        else:
            raise KeyError(f"🛑 Error Crítico: La variable objetivo '{target_col}' (o su versión '{target_norm}') no existe. Columnas vistas: {list(df.columns)}")

    print(f"=== 🧱 FASE 3.2: Aislamiento del Target ('{target_col}') y Enrutamiento ===")

    inicio_timer = time.time()

    # ==========================================
    # 1. La Regla de Oro (Purga de Target Nulo)
    # ==========================================
    filas_iniciales = len(df)

    # Copiamos y eliminamos las filas sin piedad donde el target es NaN
    df_limpio = df.dropna(subset=[target_col]).copy() 

    filas_finales = len(df_limpio)
    nulos_purgados = filas_iniciales - filas_finales

    # ==========================================
    # 2. El Aislamiento (Split X, y)
    # ==========================================
    y = df_limpio[target_col]
    X = df_limpio.drop(columns=[target_col])

    # ==========================================
    # 3. Escáner de Enrutamiento (AutoML Routing)
    # ==========================================
    rutas = {
        'num_vars': [],
        'cat_vars': [],
        'date_vars': [],
        'bool_vars': []
    }

    print("  🔍 Mapeando la topología de las características predictoras (X)...")

    for col in X.columns:
        tipo = X[col].dtype

        # Booleanas (Damos prioridad a las bool_vars para que no se confundan con numéricas)
        if pd.api.types.is_bool_dtype(tipo):
            rutas['bool_vars'].append(col)
        # Numéricas (Int y Float)
        elif pd.api.types.is_numeric_dtype(tipo):
            rutas['num_vars'].append(col)
        # Categóricas y Textos
        elif isinstance(tipo, pd.CategoricalDtype) or pd.api.types.is_object_dtype(tipo) or pd.api.types.is_string_dtype(tipo):
            rutas['cat_vars'].append(col)
        # Fechas y Tiempos
        elif pd.api.types.is_datetime64_any_dtype(tipo):
            rutas['date_vars'].append(col)
        else:
            print(f"    ⚠️ Advertencia: Tipo de dato no reconocido en '{col}': {tipo}")

    # ==========================================
    # 4. Reporte Ejecutivo de MLOps
    # ==========================================
    tiempo_total = time.time() - inicio_timer

    print(f"\n✅ Aislamiento y Enrutamiento Completado en {tiempo_total:.3f}s:")
    if nulos_purgados > 0:
        print(f"  🔪 Regla de Oro Aplicada: {nulos_purgados} filas destruidas por no tener Target.")
    else:
        print(f"  ✔️ Regla de Oro        : 0 filas destruidas (Target 100% íntegro).")

    print(f"  📦 Matriz Predictora (X): {X.shape[0]:,} filas x {X.shape[1]} columnas")
    print(f"  🎯 Vector Objetivo (y)  : {len(y):,} etiquetas aisladas")
    print(f"\n  🛣️ Mapas de Ruteo Creados:")
    print(f"    - Numéricas   (num_vars) : {len(rutas['num_vars'])} columnas")
    print(f"    - Categóricas (cat_vars) : {len(rutas['cat_vars'])} columnas")
    print(f"    - Booleanas   (bool_vars): {len(rutas['bool_vars'])} columnas")
    print(f"    - Temporales  (date_vars): {len(rutas['date_vars'])} columnas")

    return X, y, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'df_purgado' not in locals() and 'df_purgado' not in globals():
        raise EnvironmentError("No se encontró 'df_purgado'. Ejecuta la Fase 2 primero.")

    # ==========================================
    # 🧠 AUTO-DETECCIÓN DEL TARGET (Protegido)
    # ==========================================
    if 'reporte_guillotina' in locals() and reporte_guillotina.get('target_escogido'):
        variable_objetivo = reporte_guillotina['target_escogido']
        print(f"🎯 Target heredado dinámicamente: '{variable_objetivo}'\n")
    elif 'mis_targets' in locals() or 'mis_targets' in globals():
        variable_objetivo = mis_targets[0] # Fallback si no está el reporte
        print(f"🎯 Target heredado de mis_targets: '{variable_objetivo}'\n")
    else:
        raise ValueError("🛑 No se encontró la variable objetivo. Ejecuta la guillotina del Paso 1.3 primero.")

    # Generamos la división definitiva y el enrutamiento
    X, y, rutas_variables = aislar_target_y_enrutar(
        df=df_purgado, 
        target_col=variable_objetivo
    )

    # Desempaquetamos las listas para usarlas en las siguientes celdas de imputación/codificación
    num_vars = rutas_variables['num_vars']
    cat_vars = rutas_variables['cat_vars']
    bool_vars = rutas_variables['bool_vars']
    date_vars = rutas_variables['date_vars']

except Exception as e:
    print(f"🛑 Fallo inesperado en el Aislamiento del Target: {e}")


# In[44]:


num_vars


# In[45]:


cat_vars


# In[46]:


date_vars


# In[47]:


bool_vars


# In[48]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

# ==========================================
# 🚀 EL CONTENEDOR DE ESTADO (PIPELINE MANAGER)
# ==========================================
class PipelineManager:
    """
    Cerebro MLOps que transporta los datos y artefactos de una fase a otra.
    Elimina la dependencia de variables globales (locals()/globals()) y 
    hace que el código sea seguro para producción.
    """
    def __init__(self):
        # El estado centralizado
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.rutas = {
            'num_vars': [], 
            'cat_vars': [], 
            'bool_vars': [], 
            'date_vars': []
        }
        self.pesos_train = None
        self.grupos_cv = None
        # Diccionario para guardar todos los transformadores entrenados (Escaladores, Imputadores)
        self.artefactos_preprocesamiento = {} 

    def cargar_split(self, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series):
        """Inicializa las 4 matrices sagradas del Machine Learning."""
        self.X_train = X_tr
        self.X_test = X_te
        self.y_train = y_tr
        self.y_test = y_te

    def guardar_artefacto(self, nombre: str, modelo: Any):
        """Guarda un transformador entrenado en la memoria del Manager."""
        self.artefactos_preprocesamiento[nombre] = modelo


def levantar_muro_de_hierro(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    [FASE 3 - Paso 3.3] División Inmediata y Muro de Hierro.
    - Purga de Target: Aplica Regex avanzado para destruir nulos ocultos en 'y' y alinear 'X'.
    - Inteligencia (AutoML): Detecta la naturaleza del Target (y) para aplicar 
      estratificación automática si es clasificación, o corte simple si es regresión.
    - Seguridad: Verifica integridad dimensional antes del corte.
    - Arquitectura MLOps: Prepara el terreno para la regla sagrada: 
      Transformadores usarán .fit_transform() en Train y SOLO .transform() en Test.
    """
    if X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: La matriz X o el vector y están vacíos.")

    if len(X) != len(y):
        raise ValueError(f"🛑 Desalineación Crítica: X tiene {len(X)} filas pero y tiene {len(y)}.")

    print(f"=== 🧱 FASE 3.3: Levantando el Muro de Hierro (Split {100-test_size*100:.0f}/{test_size*100:.0f}) ===")

    inicio_timer = time.time()

    # ==========================================
    # 0.5. Purga de Nulos Ocultos en el Target (El Escudo Regex)
    # ==========================================
    # Usamos tu regex convirtiendo temporalmente a string para evitar errores si 'y' es numérica
    patron_regex = r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|)$|^[^a-zA-Z0-9]+$'
    mascara_regex = y.astype(str).str.match(patron_regex, na=True)

    # Combinamos con los NaNs nativos de Pandas por si acaso
    mascara_nulos_reales = y.isna()
    mascara_borrar = mascara_regex | mascara_nulos_reales

    filas_a_borrar = mascara_borrar.sum()

    if filas_a_borrar > 0:
        print(f"  🧹 [PURGA TARGET] Detectados {filas_a_borrar} registros con respuesta (y) nula/inválida.")
        # Filtramos 'y' y luego usamos sus índices sobrevivientes para filtrar 'X'
        y = y[~mascara_borrar].copy()
        X = X.loc[y.index].copy()
        print(f"  🗑️ Filas eliminadas de X e y para mantener integridad dimensional (Dataset restante: {len(y):,}).")

        if len(y) == 0:
            raise ValueError("🛑 Error Fatal: El dataset quedó vacío tras purgar los Targets inválidos.")
    else:
        print("  ✨ [TARGET LIMPIO] No se detectaron nulos ocultos en la variable objetivo.")

    # ==========================================
    # 1. Detección Inteligente de Estratificación
    # ==========================================
    # Si 'y' tiene pocos valores únicos (ej. < 100) o es texto/categoría, asumimos CLASIFICACIÓN.
    es_clasificacion = False
    if pd.api.types.is_object_dtype(y.dtype) or isinstance(y.dtype, pd.CategoricalDtype):
        es_clasificacion = True
    elif pd.api.types.is_numeric_dtype(y.dtype) and y.nunique() < 100:
        es_clasificacion = True

    estrategia_stratify = y if es_clasificacion else None

    # ==========================================
    # 2. La División (La Guillotina Temporal)
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=estrategia_stratify
    )

    # ==========================================
    # 3. Reporte de Arquitectura
    # ==========================================
    tiempo_total = time.time() - inicio_timer

    print(f"\n✅ Muro de Hierro levantado en {tiempo_total:.3f}s:")
    print(f"  🧠 Naturaleza del Target : {'Clasificación (Estratificado)' if es_clasificacion else 'Regresión (Corte Simple)'}")
    print(f"  🚂 Matriz de TRAIN       : {X_train.shape[0]:,} filas ({len(X_train)/len(X):.1%})")
    print(f"  🔒 Matriz de TEST        : {X_test.shape[0]:,} filas ({len(X_test)/len(X):.1%})")

    print("\n📜 EDICTO DE MLOPS (Regla de Oro para las Fases 4 y 5):")
    print("  1. Todo Imputador, Scaler o Encoder debe ENTRENARSE estrictamente sobre TRAIN usando .fit()")
    print("  2. TEST es ciego. Solo se le aplicará .transform() usando las reglas aprendidas de TRAIN.")

    return X_train, X_test, y_train, y_test


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'X' not in locals() or 'y' not in locals():
        raise EnvironmentError("No se encontraron 'X' e 'y'. Ejecuta el Aislamiento (Paso 3.2) primero.")

    # 1. Instanciamos el Cerebro MLOps por primera vez en el Notebook
    manager = PipelineManager()

    # 2. Ejecutamos la división de los datos crudos
    X_tr, X_te, y_tr, y_te = levantar_muro_de_hierro(
        X=X, 
        y=y, 
        test_size=0.20,
        random_state=42 
    )

    # 3. Guardamos los resultados dentro del manager para que viaje a las siguientes fases
    manager.cargar_split(X_tr, X_te, y_tr, y_te)
    print("\n💾 Matrices guardadas exitosamente en el PipelineManager.")

    # (Transición): Reflejamos en variables globales por si tus celdas de abajo aún las piden
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    y_test = manager.y_test

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en el Train/Test Split: {e}")


# In[49]:


X_train.head()


# In[50]:


X_test.head()


# In[51]:


X_train.info()


# In[52]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold

def definir_estrategia_validacion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    date_vars: Optional[List[str]] = None
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    [FASE 3 - Paso 3.4] Esquema de Validación Inteligente (Árbitro Topológico).
    - Analiza la presencia simultánea o individual de Tiempo (Fechas) e Identidad (IDs).
    - Elige matemáticamente la estrategia de K-Folds más segura para evitar Fugas de Datos.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("🛑 Error Crítico: X_train o y_train están vacíos.")

    print(f"=== 🧭 FASE 3.4: Motor de Estrategia de Validación (Cross-Validation) ===")

    inicio_timer = time.time()
    date_vars = date_vars or []
    es_clasificacion = False
    grupos_cv = None

    # ==========================================
    # 1. Detección de Naturaleza del Problema (Target)
    # ==========================================
    if pd.api.types.is_object_dtype(y_train.dtype) or isinstance(y_train.dtype, pd.CategoricalDtype):
        es_clasificacion = True
    elif pd.api.types.is_numeric_dtype(y_train.dtype) and y_train.nunique() < 100:
        es_clasificacion = True

    # ==========================================
    # 2. Extracción de Grupos (Desde el Índice)
    # ==========================================
    nombre_indice = X_train.index.name
    tiene_ids_reales = isinstance(X_train.index, pd.MultiIndex) or (nombre_indice is not None and nombre_indice != 'auto_id')

    if tiene_ids_reales:
        if isinstance(X_train.index, pd.MultiIndex):
            grupos_cv = np.array(['_'.join(map(str, idx)) for idx in X_train.index])
        else:
            grupos_cv = X_train.index.to_numpy()

    # ==========================================
    # 3. El Árbitro Inteligente (Matriz de Decisión MLOps)
    # ==========================================
    estrategia_cv = None
    nombre_estrategia = ""
    tiene_tiempo = len(date_vars) > 0

    print("  🔍 Analizando topología de la matriz para Validación Cruzada...")

    # Escenario A: Datos de Panel (Tiempo + Identidad)
    if tiene_tiempo and tiene_ids_reales:
        # Sklearn no tiene un "GroupTimeSeriesSplit" nativo perfecto, usamos GroupKFold como la opción más segura 
        # para evitar que un mismo ID se filtre entre Folds, asumiendo que los Lags (Fase 14.1) ya encapsularon la historia.
        estrategia_cv = GroupKFold(n_splits=n_splits)
        nombre_estrategia = f"GroupKFold (Prioridad ID sobre Tiempo - {len(np.unique(grupos_cv))} grupos)"
        print("    ⚠️ Conflicto detectado: La matriz tiene TIEMPO y tiene IDs simultáneamente (Datos de Panel).")
        print("    ↳ Decisión Arquitectónica: Predomina el ID. Es más crítico evitar que el modelo memorice")
        print("      el futuro de un mismo paciente/ciudad. Se usará Agrupación por Identidad.")

    # Escenario B: Serie de Tiempo Pura (Solo Tiempo, un solo protagonista)
    elif tiene_tiempo and not tiene_ids_reales:
        estrategia_cv = TimeSeriesSplit(n_splits=n_splits)
        nombre_estrategia = "TimeSeriesSplit (Corte Secuencial Histórico)"
        print("    ⏱️ Solo hay Tiempo (Sin IDs múltiples). Activando validación secuencial estricta.")

    # Escenario C: Transversal Múltiple (Solo IDs, sin reloj)
    elif not tiene_tiempo and tiene_ids_reales:
        estrategia_cv = GroupKFold(n_splits=n_splits)
        nombre_estrategia = f"GroupKFold (Agrupado estricto por Índice - {len(np.unique(grupos_cv))} grupos)"
        print("    🧬 Solo hay IDs (Sin reloj). Activando blindaje de identidad transversal.")

    # Escenario D: Transversal Simple (Ni Tiempo, Ni IDs complejos)
    else:
        if es_clasificacion:
            estrategia_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            nombre_estrategia = "StratifiedKFold (Mantiene proporción real de clases)"
            print("    ⚖️ Matriz transversal simple (Clasificación). Activando estratificación equilibrada.")
        else:
            estrategia_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            nombre_estrategia = "Standard KFold (Corte Aleatorio Simple)"
            print("    📈 Matriz transversal simple (Regresión). Activando partición aleatoria.")

    # ==========================================
    # 4. Reporte Ejecutivo
    # ==========================================
    print(f"\n✅ Estrategia de Validación Definida en {time.time() - inicio_timer:.3f}s:")
    print(f"  🎯 Esquema Final  : {nombre_estrategia}")
    print(f"  🔪 Folds (Cortes) : {n_splits}")

    if grupos_cv is not None:
        print("  ⚠️ IMPORTANTE     : El motor ha devuelto el vector 'grupos_cv'. Asegúrate de inyectarlo en tu modelo.")

    return estrategia_cv, grupos_cv


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    # 1. Validación limpia de MLOps
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta el Muro de Hierro (Paso 3.3) primero.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene datos de entrenamiento (X_train/y_train).")

    # 2. Extraemos las fechas detectadas desde las rutas del manager
    fechas_detectadas = manager.rutas.get('date_vars', [])

    # 3. Generamos la estrategia y extraemos los grupos ocultos
    cv_strategy, grupos_cv_extraidos = definir_estrategia_validacion(
        X_train=manager.X_train,
        y_train=manager.y_train,
        n_splits=5,
        date_vars=fechas_detectadas
    )

    # 4. Guardamos los activos generados en el Manager
    manager.grupos_cv = grupos_cv_extraidos
    manager.guardar_artefacto('cv_strategy', cv_strategy)
    print("\n💾 Mapa topológico de validación (Folds) guardado en el PipelineManager.")

    # (Transición): Reflejamos en variables globales por si tus celdas de abajo aún las piden
    estrategia_cv = cv_strategy
    grupos_cv = manager.grupos_cv

except Exception as e:
    print(f"🛑 Fallo inesperado en la Definición de Estrategia CV: {e}")


# In[53]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import time
from typing import Tuple, Optional

def desanclar_indice_estructural(
    X: pd.DataFrame, 
    y: Optional[pd.Series] = None,
    nombre_dataset: str = "Matriz"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    [FASE 1 - Paso 3.5] Desanclaje del Índice y Purificación.
    - Purga de Identidad: Elimina los IDs del índice para liberar memoria y evitar 
      que transformadores de Scikit-Learn/Categorical Encoders fallen por desalineación.
    - Sincronización Estricta: Resetea (X, y) en paralelo garantizando la topología.
    - Arquitectura Modular: Se ejecuta por separado para Train y Test.
    """
    if X is None or X.empty:
        raise ValueError(f"🛑 Error Crítico: La matriz X ({nombre_dataset}) está vacía o es inválida.")

    print(f"=== ⚓ FASE 3.5: Desanclaje del Índice y Sincronización ({nombre_dataset}) ===")

    inicio_timer = time.time()

    # 1. Captura de metadatos para el reporte
    nombres_indices = X.index.names

    # 2. Desanclaje de X
    X_clean = X.reset_index(drop=True)
    y_clean = None

    # 3. Desanclaje de y (Si existe) y Auditoría
    if y is not None:
        if y.empty:
            raise ValueError(f"🛑 Error Crítico: La variable objetivo y ({nombre_dataset}) está vacía.")

        y_clean = y.reset_index(drop=True)
        assert len(X_clean) == len(y_clean), f"🚨 Ruptura dimensional en {nombre_dataset} detectada tras desanclaje."

    # ==========================================
    # Reporte Ejecutivo de MLOps
    # ==========================================
    tiempo_total = time.time() - inicio_timer

    print(f"  ✅ Completado en {tiempo_total:.4f}s")
    if nombres_indices and nombres_indices[0] is not None:
        print(f"  🗑️ Índices destruidos : {list(nombres_indices)}")
    else:
        print(f"  🗑️ Índices destruidos : Índice numérico nativo")

    if y is not None:
        print(f"  🔗 Sincronización     : X e y ({len(X_clean)} filas) perfectamente alineados.")
    else:
        print(f"  🔗 Sincronización     : X ({len(X_clean)} filas) desanclada de forma independiente.")

    return X_clean, y_clean


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    # 1. Validación limpia de MLOps (Adiós a los locals() flotantes)
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene las matrices cargadas. Revisa la división Train/Test.")

    # 2. Desanclamos TRAIN extrayendo los datos del Manager
    X_train_clean, y_train_clean = desanclar_indice_estructural(
        X=manager.X_train, 
        y=manager.y_train,
        nombre_dataset="TRAIN"
    )

    # 3. Desanclamos TEST extrayendo los datos del Manager
    X_test_clean, y_test_clean = desanclar_indice_estructural(
        X=manager.X_test, 
        y=manager.y_test,
        nombre_dataset="TEST"
    )

    # 4. Guardamos los resultados purificados de vuelta en el Manager
    manager.X_train = X_train_clean
    manager.y_train = y_train_clean
    manager.X_test = X_test_clean
    manager.y_test = y_test_clean

    print("\n⚙️ Estado Global: Matrices purificadas devueltas al Manager y listas para Scikit-Learn.")

    # (Transición): Reflejamos en variables globales por si tus celdas de abajo aún las piden
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    y_test = manager.y_test

except EnvironmentError as env_err:
    print(f"⚠️ Dependencia faltante:\n{env_err}")
except AssertionError as assert_err:
    print(f"🛑 Falla Crítica de Integridad Matemática: {assert_err}")
except Exception as e:
    print(f"🛑 Fallo inesperado en el Desanclaje: {e}")


# # FASE 2: Exploración Visual y Pre-procesamiento de Texto
# Entendemos el negocio, limpiamos el ruido obvio y preparamos las categorías.

# In[54]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt # El único import correcto para gráficos
import seaborn as sns
import time
import warnings

def diagnosticar_balance_target(y: pd.Series, nombre_target: str = "Target"):
    """
    [FASE 2 - Paso 4.1] Radiografía Estadística y Diagnóstico del Target.
    - IA Analítica: Detecta automáticamente si el problema es Regresión, Binario o Multiclase.
    - Diagnóstico MLOps: Calcula el Imbalance Ratio (IR) y emite alertas estratégicas adaptadas al tipo.
    - Extracción Automática: Retorna la clase minoritaria para ser usada en fases posteriores.
    """
    if y is None or y.empty:
        raise ValueError("🛑 Error Crítico: El vector objetivo (y) está vacío o no existe.")

    print(f"=== 📊 FASE 4.1: Diagnóstico MLOps del Target ('{nombre_target}') ===")
    inicio_timer = time.time()

    # Configuración estética profesional
    sns.set_theme(style="whitegrid", palette="muted")
    warnings.simplefilter("ignore", category=FutureWarning)

    # Variable para guardar la clase que retornaremos
    clase_minoritaria_detectada = None

    # ==========================================
    # 1. Detección Inteligente de la Topología
    # ==========================================
    es_clasificacion = False

    # Soporte ultra-robusto para inferir clasificación
    if pd.api.types.is_object_dtype(y.dtype) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y.dtype):
        es_clasificacion = True
    elif pd.api.types.is_numeric_dtype(y.dtype) and y.nunique(dropna=True) <= 20: # Si es número pero tiene pocas opciones, es clasificación
        es_clasificacion = True

    plt.figure(figsize=(10, 5), dpi=100)

    # ==========================================
    # 2.A. Ruta AutoML para CLASIFICACIÓN
    # ==========================================
    if es_clasificacion:
        conteo = y.value_counts(dropna=True)
        porcentajes = y.value_counts(normalize=True, dropna=True) * 100

        # Extraemos la clase con menos registros sin importar el tipo
        clase_minoritaria_detectada = conteo.index[-1]

        # --- BIFURCACIÓN 1: BINARIO EXACTO (2 CLASES) ---
        if len(conteo) == 2:
            clase_mayoritaria = conteo.index[0]
            imbalance_ratio = conteo.iloc[0] / conteo.iloc[-1]

            print(f"  🧠 Naturaleza Detectada: Clasificación Binaria (2 clases)")
            print(f"  📈 Clase Mayoritaria   : '{clase_mayoritaria}' ({porcentajes.iloc[0]:.1f}%)")
            print(f"  📉 Clase Minoritaria   : '{clase_minoritaria_detectada}' ({porcentajes.iloc[-1]:.1f}%)")
            print(f"  ⚖️ Imbalance Ratio (IR): 1:{imbalance_ratio:.2f} (Por cada minoría hay {imbalance_ratio:.1f} mayorías)")

            print("\n  🚨 DIAGNÓSTICO ESTRATÉGICO PARA FASE 6:")
            if imbalance_ratio > 9: 
                print("     [ALERTA ROJA] Desbalance Severo. Requisito obligatorio: Aplicar SMOTE o Class Weights extremos.")
            elif imbalance_ratio > 3: 
                print("     [ALERTA AMARILLA] Desbalance Moderado. Se sugiere activar Class Weights en LightGBM/XGBoost.")
            else:
                print("     [VERDE] Balance Aceptable. No se requieren técnicas de sobre-muestreo.")

        # --- BIFURCACIÓN 2: MULTICLASE (3 O MÁS CLASES) ---
        elif len(conteo) >= 3:
            clase_dominante = conteo.index[0]
            imbalance_ratio_extremo = conteo.iloc[0] / conteo.iloc[-1]

            print(f"  🧠 Naturaleza Detectada: Clasificación Multiclase ({len(conteo)} clases únicas)")
            print(f"  👑 Clase Dominante     : '{clase_dominante}' ({porcentajes.iloc[0]:.1f}%)")
            print(f"  ⚠️ Clase más débil     : '{clase_minoritaria_detectada}' ({porcentajes.iloc[-1]:.1f}%)")
            print(f"  ⚖️ IR Extremo (Max/Min): 1:{imbalance_ratio_extremo:.2f} (Brecha entre el mayor y el menor)")

            print("\n  🚨 DIAGNÓSTICO ESTRATÉGICO MULTICLASE PARA FASE 6:")
            if imbalance_ratio_extremo > 9: 
                print("     [ALERTA ROJA] Desbalance Severo Multiclase. La clase más débil está casi extinta.")
                print("                   Requisito: SMOTE Multiclase o pesos de clase balanceados (class_weight='balanced').")
            elif imbalance_ratio_extremo > 3: 
                print("     [ALERTA AMARILLA] Desbalance Moderado. Ciertas clases tienen poca representación.")
                print("                       Sugerencia: Evaluar el modelo usando la métrica 'F1-Macro' en lugar de Accuracy.")
            else:
                print("     [VERDE] Balance Aceptable. Las clases están distribuidas de forma segura.")

        # --- BIFURCACIÓN 3: ERROR DE VARIANZA CERO ---
        else:
            print("  🛑 [ERROR CRÍTICO] Target de una sola clase (Varianza Cero). El modelo no puede aprender a discriminar.")

        # 🚀 FIX MLOps: Forzamos el 'order' para que Seaborn dibuje de mayor a menor frecuencia
        ax = sns.barplot(x=conteo.index, y=conteo.values, order=conteo.index, edgecolor=".2")
        plt.title(f"Distribución de Clases: {nombre_target}", fontsize=14, pad=15)
        plt.ylabel("Frecuencia (N° de filas)")

        # 🚀 FIX MLOps: Cálculo matemático directo. Extraemos la altura de la barra dibujada 
        # y calculamos el % en tiempo real. Cero posibilidad de desfase.
        total_filas = len(y.dropna())
        for p in ax.patches:
            altura_barra = p.get_height()
            pct_real = (altura_barra / total_filas) * 100

            ax.annotate(f'{pct_real:.1f}%', 
                        (p.get_x() + p.get_width() / 2., altura_barra), 
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), 
                        textcoords='offset points')

    # ==========================================
    # 2.B. Ruta AutoML para REGRESIÓN
    # ==========================================
    else:
        media = y.mean()
        mediana = y.median()
        sesgo = y.skew()

        print(f"  🧠 Naturaleza Detectada: Regresión (Valores continuos)")
        print(f"  📏 Media   : {media:.2f}")
        print(f"  📍 Mediana : {mediana:.2f}")
        print(f"  📐 Sesgo   : {sesgo:.2f}")

        # Histograma con curva de densidad (KDE)
        sns.histplot(y, kde=True, bins=50, color='steelblue')
        plt.axvline(media, color='red', linestyle='--', label=f'Media: {media:.2f}')
        plt.axvline(mediana, color='green', linestyle='-', label=f'Mediana: {mediana:.2f}')
        plt.title(f"Distribución Continua: {nombre_target}", fontsize=14, pad=15)
        plt.legend()

        print("\n  🚨 DIAGNÓSTICO ESTRATÉGICO PARA FASE 6:")
        if abs(sesgo) > 1:
            print("     [ALERTA AMARILLA] Cola pesada detectada (Sesgo alto). Se sugiere evaluar Log-Transform (np.log1p) antes de entrenar.")
        else:
            print("     [VERDE] Distribución simétrica aceptable.")

    # ==========================================
    # 3. Finalización
    # ==========================================
    plt.tight_layout()
    plt.show()
    print(f"\n⏱️ Diagnóstico completado en {time.time() - inicio_timer:.3f}s")

    # Retornamos la clase minoritaria al entorno
    return clase_minoritaria_detectada


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.y_train is None:
        raise ValueError("El Manager no tiene cargado el vector 'y_train'.")

    # AUTO-DETECCIÓN DEL NOMBRE DEL TARGET (De los pasos anteriores, o por defecto)
    nombre_target_heredado = manager.y_train.name if manager.y_train.name else "Target_Manager"

    # Ejecutamos la radiografía usando el target almacenado en el manager
    clase_minoritaria_global = diagnosticar_balance_target(
        y=manager.y_train, 
        nombre_target=nombre_target_heredado 
    )

    if clase_minoritaria_global is not None:
        # Guardamos la clase minoritaria en la memoria del manager (rutas) para usarla en el futuro
        manager.rutas['clase_minoritaria'] = clase_minoritaria_global
        print(f"📦 [MLOps] Clase minoritaria '{clase_minoritaria_global}' capturada en el Manager y lista para Fase 4.3.")

except Exception as e:
    print(f"🛑 Error en el Diagnóstico: {e}")


# In[55]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import re
from IPython.display import display, Markdown

def auditar_logica_dominio_integral(X: pd.DataFrame, y: pd.Series = None):
    """
    [FASE 2 - Paso 4.2] Escáner AutoML Multidimensional Agolnóstico de Anomalías de Distribución.
    - Arquitectura Modular: Analiza Numéricos, Categóricos, Fechas y Booleanos independientemente de los nombres de columnas.
    - Inteligencia Estadística: Detecta valores imposibles (ej. negativos donde no debe), 
      outliers extremos (fences), varianza cero y fechas huérfanas/futuras.
    - Visualización Térmica: Aplica Heatmap de degradado al % de dominancia (Moda).
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz de características (X) está vacía.")

    print(f"=== 🔬 FASE 4.2: Auditoría Integral y Mapa de Calor de Varianza (Agnóstico) ===")
    inicio_timer = time.time()
    alertas_totales = 0
    diccionario_resultados = {} 

    # Ensamblaje temporal blindado de X e y
    df_analisis = X.copy()
    if y is not None:
        target_name = y.name if y.name else 'Target_y'
        # Evitamos colisiones de nombres
        if target_name in df_analisis.columns: target_name = f"{target_name}_TargetInyectado"
        df_analisis[target_name] = y

    total_filas = len(df_analisis)

    # ==========================================
    # Funciones Auxiliares de Visualización (Intactas)
    # ==========================================
    def inyectar_moda_y_ordenar(df_stats, cols, df_origen):
        tops, freqs = [], []
        for c in cols:
            vc = df_origen[c].value_counts(dropna=True)
            if not vc.empty:
                tops.append(vc.index[0])
                freqs.append(vc.iloc[0])
            else:
                tops.append(np.nan); freqs.append(0)
        if 'top' not in df_stats.columns: df_stats['top'] = tops
        if 'freq' not in df_stats.columns: df_stats['freq'] = freqs
        df_stats['% Moda'] = (df_stats['freq'].astype(float) / total_filas) * 100
        return df_stats.sort_values(by='% Moda', ascending=False)

    def mostrar_tabla_con_gradiente(df_stats):
        estilo = df_stats.style.background_gradient(
            subset=['% Moda'], cmap='YlOrRd'
        ).format({'% Moda': '{:.2f}%', 'freq': '{:.0f}'})
        display(estilo)

    # ==========================================
    # 1. Análisis de Variables Numéricas (Inteligencia Híbrida)
    # ==========================================
    num_cols = df_analisis.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        display(Markdown("### 🔢 1. Variables Numéricas (Anomalías de Signo y Outliers Extremos)"))
        stats_num = df_analisis[num_cols].describe().T
        stats_num = inyectar_moda_y_ordenar(stats_num, num_cols, df_analisis)
        diccionario_resultados['Numericas'] = stats_num

        for col in stats_num.index:
            min_val = stats_num.loc[col, 'min']
            max_val = stats_num.loc[col, 'max']
            std_val = stats_num.loc[col, 'std']
            mean_val = stats_num.loc[col, 'mean']
            q3_val = stats_num.loc[col, '75%']
            q1_val = stats_num.loc[col, '25%']

            # --- 1. Signos Imposibles ---
            if min_val < 0:
                print(f"  🚨 [SIGNO NEGATIVO] '{col}': Contiene valores negativos ({min_val}). Verificar lógica de negocio.")
                alertas_totales += 1

            # --- 2. Varianza Cero ---
            if std_val == 0:
                print(f"  🧊 [VARIANZA CERO] '{col}': Todos los valores son idénticos. Inútil para predictivo.")
                alertas_totales += 1

            # --- 3. Outliers Estadísticos Extremos (IQR) ---
            iqr = q3_val - q1_val
            techo_iqr = q3_val + (3 * iqr)

            if iqr > 0 and max_val > techo_iqr and max_val > 100:
                 print(f"  🔥 [OUTLIER IQR] '{col}': Máximo ({max_val}) rompe el techo estadístico IQR ({techo_iqr:.2f}).")
                 alertas_totales += 1

            # --- 4. Anomalías Gravitacionales (Para distribuciones dominadas por ceros donde IQR=0) ---
            elif std_val > 0 and max_val > (mean_val + (10 * std_val)):
                 print(f"  ☄️ [ANOMALÍA GRAVITACIONAL] '{col}': Máximo ({max_val}) está a >10 desviaciones estándar de la media. Extremo absurdo.")
                 alertas_totales += 1

            # --- 5. Códigos Legacy Universales (El detector de 9999s) ---
            # Convierte el número a entero (para ignorar decimales) y busca si empieza con tres o más nueves
            if max_val >= 999 and re.match(r'^9{3,}', str(int(max_val))):
                 print(f"  🤖 [CÓDIGO SISTEMA LEGACY] '{col}': Máximo ({max_val}) parece un NaN codificado por sistemas antiguos (999...).")
                 alertas_totales += 1

        mostrar_tabla_con_gradiente(stats_num)
        print("-" * 80)

    # ==========================================
    # 2. Variables Categóricas (Ruido y Máscaras Regex - Intactas/Genéricas)
    # ==========================================
    cat_cols = df_analisis.select_dtypes(include=['object', 'category', 'string']).columns
    if len(cat_cols) > 0:
        display(Markdown("### 🔠 2. Variables Categóricas (Ruido y Máscaras Regex)"))
        stats_cat = df_analisis[cat_cols].describe().T
        stats_cat = inyectar_moda_y_ordenar(stats_cat, cat_cols, df_analisis)
        diccionario_resultados['Categoricas'] = stats_cat

        patron_mascara = re.compile(r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|)$|^[^a-zA-Z0-9]+$')

        for col in stats_cat.index:
            unicos = stats_cat.loc[col, 'unique']

            if unicos > (total_filas * 0.9):
                print(f"  🌪️ [RUIDO ABSOLUTO] '{col}': {unicos} valores únicos. Actúa como un ID basura.")
                alertas_totales += 1

            valores_distintos = df_analisis[col].dropna().astype(str).unique()
            mascaras_encontradas = [val for val in valores_distintos if patron_mascara.match(val.strip())]

            if mascaras_encontradas:
                ejemplos = ", ".join(f"'{m}'" for m in mascaras_encontradas[:3])
                print(f"  🎭 [MÁSCARA DETECTADA] '{col}': Contiene nulos camuflados ({ejemplos}).")
                alertas_totales += 1

        mostrar_tabla_con_gradiente(stats_cat)
        print("-" * 80)

    # ==========================================
    # 3. Variables Booleanas (Desbalance - Intactas/Genéricas)
    # ==========================================
    bool_cols = df_analisis.select_dtypes(include=['bool', 'boolean']).columns
    if len(bool_cols) > 0:
        display(Markdown("### ⚖️ 3. Variables Booleanas (Desbalance Extremo)"))
        # Los booleanos modernos (boolean) de pandas necesitan un casteo temporal a string para describe()
        stats_bool = df_analisis[bool_cols].astype(str).describe().T
        stats_bool = inyectar_moda_y_ordenar(stats_bool, bool_cols, df_analisis)
        diccionario_resultados['Booleanas'] = stats_bool

        for col in stats_bool.index:
            porcentaje_top = stats_bool.loc[col, '% Moda']
            if porcentaje_top > 99.0:
                print(f"  🧊 [VARIANZA CONGELADA] '{col}': El {porcentaje_top:.1f}% es '{stats_bool.loc[col, 'top']}'. Inútil.")
                alertas_totales += 1

        mostrar_tabla_con_gradiente(stats_bool)
        print("-" * 80)

    # ==========================================
    # 4. Variables de Fecha (Viajes en el Tiempo)
    # ==========================================
    date_cols = df_analisis.select_dtypes(include=['datetime', 'datetimetz']).columns
    if len(date_cols) > 0:
        display(Markdown("### 📅 4. Variables de Fecha (Viajes en el Tiempo)"))

        # 🔧 FIX APLICADO: Eliminado datetime_is_numeric=True para compatibilidad con Pandas >= 2.0
        stats_date = df_analisis[date_cols].describe().T
        stats_date = inyectar_moda_y_ordenar(stats_date, date_cols, df_analisis)
        diccionario_resultados['Fechas'] = stats_date

        fecha_actual = pd.Timestamp.now()
        fecha_pivote_antigua = pd.Timestamp('1900-01-01')

        for col in stats_date.index:
            min_date, max_date = stats_date.loc[col, 'min'], stats_date.loc[col, 'max']

            if max_date > fecha_actual:
                print(f"  🚀 [VIAJE AL FUTURO] '{col}': Fecha máxima ({max_date.date()}) es mayor a hoy.")
                alertas_totales += 1

            if min_date <= fecha_pivote_antigua:
                print(f"  🦖 [FECHA FÓSIL] '{col}': Fecha mínima ({min_date.date()}). Sospecha de error.")
                alertas_totales += 1

        mostrar_tabla_con_gradiente(stats_date)
        print("-" * 80)

    # ==========================================
    # 5. Reporte Ejecutivo
    # ==========================================
    if alertas_totales == 0:
        display(Markdown("### ✅ [DOMINIO COMPLETAMENTE LIMPIO] Ninguna anomalía de distribución detectada."))
    else:
        display(Markdown(f"### 🛑 Total de Anomalías Inter-Dominio Detectadas: **{alertas_totales}**"))
        print("💡 Acción: Utiliza esta información para el Paso 8 (Desenmascarar Nulos) y el Paso 12 (Outliers).")

    print(f"\n⏱️ Auditoría Integral completada en {time.time() - inicio_timer:.3f}s")

    return diccionario_resultados

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'y_train'.")

    # Ejecutar la sonda multiespectral agnóstica extrayendo los datos del manager
    dicc_describe = auditar_logica_dominio_integral(X=manager.X_train, y=manager.y_train)

except Exception as e:
    print(f"🛑 Error en la Validación Integral: {e}")


# # Solo se usa si el Dataset involucra al Humano

# In[56]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def auditar_sesgo_historico(
    X: pd.DataFrame, 
    y: pd.Series, 
    clase_favorable=None, 
    columnas_sensibles=None
):
    """
    [FASE 2 - Paso 4.3] Motor AutoML de Justicia Algorítmica (Fairness).
    - Escáner Legal Contextual: Detecta atributos protegidos usando delimitadores de bases de datos.
    - Soporte Datetime (NUEVO): Detecta fechas de nacimiento y extrae el año automáticamente.
    - Discretización Inteligente: Convierte variables continuas (como edad/año) en rangos generacionales.
    - Calcula la Tasa de Aprobación Base por grupo sociodemográfico.
    - Aplica la regla legal del 80% (Disparate Impact Ratio).
    """
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: Las matrices X_train o y_train están vacías.")

    print(f"=== ⚖️ FASE 4.3: Auditoría de Atributos Protegidos (Línea Base de Sesgo) ===")
    inicio_timer = time.time()

    # 1. Ensamblaje temporal para análisis
    df_analisis = X.copy()
    target_name = y.name if y.name else 'Target'
    df_analisis[target_name] = y

    # ==========================================
    # 🚀 2. Detección Inteligente de Atributos Protegidos (Bilingüe + Contextual)
    # ==========================================
    if not columnas_sensibles:
        def construir_patron(palabra):
            return fr'(^{palabra}$|^{palabra}_|_{palabra}$|_{palabra}_|[a-z]{palabra.capitalize()})'

        # 📚 DICCIONARIO LEGAL EXHAUSTIVO (GDPR, EEOC, Leyes Latam)
        # Se usan palabras completas exactas para que el constructor de CamelCase/Snake_Case funcione perfecto.
        terminos_legales = [
            # Edad y Nacimiento / Age & Birth
            'age', 'edad', 'dob', 'birth', 'birthdate', 'nacimiento', 'year', 'año', 'ano', 'generation', 'generacion',
            # Sexo, Género y Orientación / Gender & Orientation
            'sex', 'sexo', 'gender', 'genero', 'orientation', 'orientacion', 'lgbt', 'lgbtq', 'sexual', 'female', 'male', 'mujer', 'hombre', 'trans', 'transgender',
            # Raza, Etnia y Color / Race, Ethnicity & Color
            'race', 'raza', 'ethnic', 'ethnicity', 'etnia', 'origen', 'color', 'minority', 'minoria', 'hispanic', 'hispano', 'latino', 'afro', 'afroamerican', 'black', 'white', 'asian', 'indigenous', 'indigena', 'tribe', 'tribu', 'caucasian',
            # Religión y Fe / Religion & Beliefs
            'religion', 'belief', 'creencia', 'faith', 'fe', 'worship', 'culto', 'muslim', 'jewish', 'christian', 'catholic', 'islam', 'judaismo', 'cristianismo',
            # Nacionalidad e Inmigración / Nationality & Immigration
            'nationality', 'nacionalidad', 'nation', 'nacion', 'country', 'pais', 'citizen', 'citizenship', 'ciudadano', 'ciudadania', 'immigration', 'inmigracion', 'migrant', 'migrante', 'native', 'nativo', 'alien',
            # Discapacidad, Salud y Maternidad / Disability, Health & Maternity
            'disability', 'discapacidad', 'handicap', 'health', 'salud', 'disease', 'enfermedad', 'illness', 'condition', 'condicion', 'pregnant', 'pregnancy', 'embarazo', 'maternity', 'maternidad',
            # Estado Civil y Familia / Marital Status & Family
            'marital', 'civil', 'marriage', 'matrimonio', 'wedding', 'spouse', 'esposo', 'esposa', 'family', 'familia', 'children', 'hijos', 'dependent', 'dependents', 'dependiente', 'dependientes', 'status', 'estado',
            # Socioeconómico (Proxies de sesgo) / Socioeconomic
            'income', 'ingreso', 'salary', 'salario', 'poverty', 'pobreza', 'education', 'educacion'
        ]

        patrones_completos = [construir_patron(t) for t in terminos_legales]
        patron_sensible = re.compile('|'.join(patrones_completos), re.IGNORECASE)

        columnas_sensibles = [col for col in X.columns if patron_sensible.search(col)]

    if not columnas_sensibles:
        print("  ✅ [INFO] No se detectaron columnas sociodemográficas protegidas en la matriz.")
        return None

    print(f"  🛡️ Atributos Protegidos detectados automáticamente: {columnas_sensibles}")
    print("  🔒 ESTATUS: Aislados lógicamente. NO SERÁN ELIMINADOS de la matriz.\n")

    # 3. Detección de la Clase Favorable
    if clase_favorable is None:
        conteo_clases = y.value_counts(normalize=True)
        clase_favorable = conteo_clases.index[-1] 
        print(f"  🎯 Clase Favorable auto-detectada: '{clase_favorable}' (Representa el {conteo_clases.iloc[-1]*100:.1f}%)")
    else:
        print(f"  🎯 Clase Favorable inyectada por MLOps: '{clase_favorable}'")

    df_analisis['target_binario_fairness'] = (df_analisis[target_name] == clase_favorable).astype(int)
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 🧠 4. Motor de Medición de Disparidad (DIR) con Auto-Binning
    # ==========================================
    for col in columnas_sensibles:
        col_analisis = col

        display(Markdown(f"### 🔍 Analizando Sesgo Sociodemográfico en: `{col}`"))

        # 🚀 FIX MLOps: Interceptor de Fechas (Datetime)
        if pd.api.types.is_datetime64_any_dtype(df_analisis[col]):
            print(f"  🧠 [AutoML] Fecha detectada en '{col}'. Extrayendo el Año para análisis generacional...")
            col_analisis = f"{col}_year"
            df_analisis[col_analisis] = df_analisis[col].dt.year

        # 🚀 LA MAGIA: Si es un número continuo (edad o año extraído), lo agrupamos en rangos (cuartiles)
        if pd.api.types.is_numeric_dtype(df_analisis[col_analisis]) and df_analisis[col_analisis].nunique() > 10:
            print(f"  🧠 [AutoML] Transformando variable continua '{col_analisis}' en rangos demográficos para medir el sesgo de forma justa...")
            col_agrupada = f"{col_analisis}_rangos"
            df_analisis[col_agrupada] = pd.qcut(df_analisis[col_analisis], q=4, duplicates='drop').astype(str)
            col_analisis = col_agrupada

        conteo_val = df_analisis[col_analisis].value_counts(normalize=True)
        # Ignoramos categorías con menos del 1% para no alertar sobre valores atípicos irrelevantes
        categorias_validas = conteo_val[conteo_val > 0.01].index 
        df_filtrado = df_analisis[df_analisis[col_analisis].isin(categorias_validas)]

        tabla_tasas = df_filtrado.groupby(col_analisis)['target_binario_fairness'].agg(['mean', 'count']).reset_index()
        tabla_tasas.rename(columns={'mean': 'Tasa_Exito', 'count': 'Muestra_Total'}, inplace=True)
        tabla_tasas.sort_values(by='Tasa_Exito', ascending=False, inplace=True)

        if tabla_tasas.empty:
            print(f"  ⚠️ No hay suficientes datos consistentes en '{col}' para graficar sesgos.")
            continue

        grupo_privilegiado = tabla_tasas.iloc[0][col_analisis]
        tasa_maxima = tabla_tasas.iloc[0]['Tasa_Exito']

        # Evitamos división por cero si la tasa máxima es 0
        if tasa_maxima == 0:
            tabla_tasas['DIR (Impacto Dispar)'] = 1.0
        else:
            tabla_tasas['DIR (Impacto Dispar)'] = tabla_tasas['Tasa_Exito'] / tasa_maxima

        # Visualización
        plt.figure(figsize=(10, 4))
        ax = sns.barplot(data=tabla_tasas, x=col_analisis, y='Tasa_Exito', palette='coolwarm')
        plt.axhline(tasa_maxima * 0.8, color='red', linestyle='--', label='Límite Legal MLOps (Regla 80%)')
        plt.title(f"Tasa de obtención de '{clase_favorable}' por {col}", fontsize=14)
        plt.ylabel("Probabilidad de Éxito")
        plt.ylim(0, max(0.5, tasa_maxima + 0.1))

        # Rotamos las etiquetas si son textos largos (rangos de edad)
        plt.xticks(rotation=15) 
        plt.legend()
        plt.show()

        # ==========================================
        # Generación de Reporte y Alertas (DOBLE COLUMNA)
        # ==========================================
        print(f"  👑 Grupo Históricamente Privilegiado: '{grupo_privilegiado}' (Tasa base: {tasa_maxima*100:.1f}%)")
        print("  " + "="*85)

        alertas_sesgo = 0
        mensajes = [] 

        for _, row in tabla_tasas.iterrows():
            grupo_actual = row[col_analisis]
            dir_actual = row['DIR (Impacto Dispar)']

            if grupo_actual == grupo_privilegiado:
                continue

            if dir_actual < 0.80:
                mensajes.append(f"🚨 [ALERTA] '{grupo_actual}': DIR = {dir_actual:.2f} ({row['Tasa_Exito']*100:.1f}%)")
                alertas_sesgo += 1
            else:
                mensajes.append(f"✅ [JUSTO] '{grupo_actual}': DIR = {dir_actual:.2f} ({row['Tasa_Exito']*100:.1f}%)")

        # Motor de Paginación a 2 Columnas
        lote_size = 20
        for i in range(0, len(mensajes), lote_size):
            lote = mensajes[i:i + lote_size]
            mitad = (len(lote) + 1) // 2 

            for j in range(mitad):
                col1 = lote[j]
                col2 = lote[j + mitad] if (j + mitad) < len(lote) else ""
                print(f"  {col1:<40} |  {col2}")

            if (i + lote_size) < len(mensajes):
                print("  " + "-"*85)

        print("  " + "="*85)
        if alertas_sesgo > 0:
            print(f"  💡 ACCIÓN REQUERIDA (Fase 5): Implementar Reweighing sobre '{col}' para neutralizar el sesgo.")
        print("\n")

    print(f"⏱️ Auditoría de Atributos Protegidos completada en {time.time() - inicio_timer:.3f}s")

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargados 'X_train' o 'y_train'.")

    # Inyección Automática de la clase minoritaria descubierta, si existe en el manager
    clase_fav_dinamica = manager.rutas.get('clase_minoritaria', None)

    # El escáner legal operará al 100% de forma autónoma con límites contextuales usando los datos del manager
    auditar_sesgo_historico(
        X=manager.X_train, 
        y=manager.y_train, 
        clase_favorable=clase_fav_dinamica 
    )

except Exception as e:
    print(f"🛑 Error en la Auditoría de Sesgo: {e}")


# In[57]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

def radiografia_visual_numericas(X: pd.DataFrame, y: pd.Series):
    """
    [FASE 2 - Paso 5.1] Motor AutoML de Visualización Bivariada (Numéricas).
    - Escudo RAM (NUEVO): Submuestrea datasets masivos a 15k filas solo para renderizado visual (evita colapsos en KDE).
    - Genera un Dashboard 1x2 por cada variable numérica predictora.
    - Izquierda: Distribución (Histograma + KDE) solapada por el Target.
    - Derecha: Boxplot para análisis de Outliers y Medianas por clase.
    - Ignora variables nulas o colapsadas automáticamente para evitar crashes.
    """
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: Las matrices X_train o y_train están vacías.")

    print(f"=== 👁️ FASE 5.1: Análisis Visual Geométrico (Numéricas vs Target) ===")
    inicio_timer = time.time()

    # 🔧 FIX MLOps: Silenciadores de Consola
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Glyph.*") # 🤫 Apaga las alertas por emojis o símbolos especiales en los gráficos
    warnings.filterwarnings("ignore", module="IPython.core.pylabtools") # 🤫 Blindaje extra para Jupyter

    # 1. Configuración de Alta Legibilidad (Seaborn)
    sns.set_theme(style="whitegrid", context="notebook")
    paleta_target = "Set2" # Paleta amigable para daltonismo (Colorblind-friendly)

    # 2. Aislamiento y Ensamblaje Seguro
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("  ⚠️ [INFO] No se detectaron variables numéricas para visualizar.")
        return

    df_viz = X[num_cols].copy()
    target_name = y.name if y.name else 'Target'
    df_viz[target_name] = y

    total_filas = len(df_viz)

    # ==========================================
    # 🚀 ESCUDO RAM: Submuestreo Visual (Big Data)
    # ==========================================
    MAX_PLOT_SAMPLES = 15000
    if total_filas > MAX_PLOT_SAMPLES:
        print(f"  ⚠️ [ALERTA DE RENDIMIENTO] Matriz masiva detectada ({total_filas:,} filas).")
        print(f"  ⚡ Protegiendo RAM: Submuestreando a {MAX_PLOT_SAMPLES:,} filas aleatorias solo para renderizado...")
        df_plot = df_viz.sample(n=MAX_PLOT_SAMPLES, random_state=42)
    else:
        df_plot = df_viz

    print(f"  📊 Renderizando Dashboards para {len(num_cols)} variables numéricas...\n")

    # ==========================================
    # 3. Motor de Renderizado Iterativo
    # ==========================================
    for col in num_cols:
        # A. Filtro AutoML de Seguridad: Omitir si la varianza es cero o tiene 100% nulos
        if df_viz[col].nunique() <= 1:
            print(f"  ⏭️ Saltando '{col}': Varianza Cero detectada (Constante).")
            continue

        # B. Creación del Lienzo (Dashboard 1 fila x 2 columnas)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        fig.suptitle(f"Radiografía de: {col}", fontsize=16, fontweight='bold', y=1.05)

        # --- PANEL IZQUIERDO: Distribución (Hist + KDE) ---
        # Usamos common_norm=False para que las montañas se escalen independientemente 
        # y podamos ver la forma de la clase minoritaria sin que la mayoritaria la aplaste.
        # 🔧 FIX: Usamos df_plot (ligero) en vez de df_viz
        sns.histplot(
            data=df_plot, x=col, hue=target_name, 
            kde=True, element="step", stat="density", common_norm=False, 
            palette=paleta_target, alpha=0.4, ax=axes[0]
        )
        axes[0].set_title(f"Distribución y Densidad (KDE)", fontsize=13)
        axes[0].set_ylabel("Densidad Probabilística")
        axes[0].set_xlabel(col)

        # --- PANEL DERECHO: Boxplot (Outliers y Medianas) ---
        # 🔧 FIX: Usamos df_plot (ligero) en vez de df_viz
        sns.boxplot(
            data=df_plot, x=target_name, y=col, 
            palette=paleta_target, showmeans=True, 
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
            ax=axes[1]
        )
        axes[1].set_title(f"Caja y Bigotes (Separación de Clases)", fontsize=13)
        axes[1].set_ylabel(col)
        axes[1].set_xlabel("Target Class")

        # C. Inyección de Información Estadística Textual Rápida
        # 🔧 MANTENIDO: Aquí SÍ usamos df_viz (matriz completa) para que el cálculo matemático sea 100% real
        media_clases = df_viz.groupby(target_name)[col].mean().to_dict()
        texto_medias = " | ".join([f"Media {k}: {v:.1f}" for k, v in media_clases.items()])
        plt.figtext(0.5, -0.05, f"Estadística Exacta (100% de datos) ➔ {texto_medias}", ha="center", fontsize=11, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

        plt.tight_layout()
        plt.show()
        print("-" * 100)

    # 🔧 Restauramos las advertencias generales al finalizar
    warnings.filterwarnings("default", message=".*Glyph.*")

    print(f"⏱️ Renderizado completado en {time.time() - inicio_timer:.2f}s")


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'y_train'.")

    # Ejecutar el motor de visualización alimentándolo desde el manager
    radiografia_visual_numericas(X=manager.X_train, y=manager.y_train)

except Exception as e:
    print(f"🛑 Error en la visualización: {e}")


# In[58]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

def radiografia_visual_categoricas(
    X: pd.DataFrame, 
    y: pd.Series, 
    clase_favorable=None,
    max_cats_visual=12 # Límite inteligente para no saturar la pantalla
):
    """
    [FASE 2 - Paso 5.2] Motor AutoML de Visualización (Categóricas).
    - Escudo RAM (NUEVO): Submuestrea datasets masivos a 15k filas solo para el Countplot visual.
    - Izquierda: Frecuencia Absoluta (Volumen total segmentado por Target).
    - Derecha: Target Rate (Probabilidad de éxito), ordenado de mayor a menor (Usa 100% de los datos).
    - IA Visual: Agrupa colas largas (alta cardinalidad) en 'OTROS' para mantener legibilidad.
    - Resiliencia: Convierte NaNs explícitamente a texto para hacerlos visibles.
    """
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: Las matrices están vacías.")

    print(f"=== 🔠 FASE 5.2: Análisis Visual (Categóricas vs Target) ===")
    inicio_timer = time.time()
    warnings.simplefilter("ignore")
    sns.set_theme(style="whitegrid", context="notebook")
    paleta_target = "Set2"

    # 1. Aislar variables categóricas (Texto, Categorías y Booleanos)
    cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
    if not cat_cols:
        print("  ⚠️ [INFO] No se detectaron variables categóricas para visualizar.")
        return

    # Ensamblaje Seguro
    df_viz = X[cat_cols].copy()
    target_name = y.name if y.name else 'Target'
    df_viz[target_name] = y

    # 2. Determinar la Clase Favorable (para calcular probabilidades)
    if clase_favorable is None:
        clase_favorable = y.value_counts().index[-1]
    df_viz['target_binario'] = (df_viz[target_name] == clase_favorable).astype(int)

    total_filas = len(df_viz)

    # ==========================================
    # 🚀 ESCUDO RAM: Muestreo Único de Alta Velocidad
    # ==========================================
    MAX_PLOT_SAMPLES = 15000
    if total_filas > MAX_PLOT_SAMPLES:
        print(f"  ⚠️ [ALERTA DE RENDIMIENTO] Matriz masiva detectada ({total_filas:,} filas).")
        print(f"  ⚡ Protegiendo RAM: Extrayendo muestra de {MAX_PLOT_SAMPLES:,} filas para gráficos de volumen...")
        # Guardamos solo los índices para aplicar el filtro rápidamente dentro del bucle
        indices_muestra = df_viz.sample(n=MAX_PLOT_SAMPLES, random_state=42).index
        nota_muestreo = " (Muestra 15k)"
    else:
        indices_muestra = df_viz.index
        nota_muestreo = ""

    print(f"  📊 Renderizando Dashboards para {len(cat_cols)} variables categóricas...\n")

    # ==========================================
    # 3. Motor de Renderizado Iterativo
    # ==========================================
    for col in cat_cols:
        # A. Tratamiento de Nulos y Tipos para visualización segura
        df_viz[col] = df_viz[col].astype(str).replace('nan', 'MISSING_NaN')

        # B. Filtro AutoML de Alta Cardinalidad (Protección visual)
        unicos = df_viz[col].nunique()
        if unicos == 1:
            print(f"  ⏭️ Saltando '{col}': Constante absoluta.")
            continue

        if unicos > max_cats_visual:
            # Mantener el Top N y agrupar el resto en 'OTROS_AGRUPADOS'
            top_categorias = df_viz[col].value_counts().nlargest(max_cats_visual - 1).index
            df_viz[f"{col}_viz"] = df_viz[col].where(df_viz[col].isin(top_categorias), 'OTROS_AGRUPADOS')
            col_plot = f"{col}_viz"
        else:
            col_plot = col

        # C. Creación del Lienzo
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        fig.suptitle(f"Radiografía de: {col} (Clase Éxito: '{clase_favorable}')", fontsize=16, fontweight='bold', y=1.05)

        # --- PANEL IZQUIERDO: Volumen Absoluto (Countplot) ---
        # 🔧 FIX: Usamos el dataframe filtrado por el escudo de RAM
        df_plot = df_viz.loc[indices_muestra]
        orden_volumen = df_plot[col_plot].value_counts().index

        sns.countplot(
            data=df_plot, x=col_plot, hue=target_name, 
            order=orden_volumen, palette=paleta_target, ax=axes[0], alpha=0.9
        )
        axes[0].set_title(f"Volumen de Filas por Categoría{nota_muestreo}", fontsize=13)
        axes[0].set_ylabel("Frecuencia (Cantidad)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis='x', rotation=45)

        # --- PANEL DERECHO: Target Rate (Probabilidad de Éxito) ---
        # 🔧 MANTENIDO: Calculamos la media usando el 100% de los datos (df_viz) porque groupby es ultra-rápido
        tasa_exito = df_viz.groupby(col_plot)['target_binario'].mean().sort_values(ascending=False).reset_index()

        sns.barplot(
            data=tasa_exito, x=col_plot, y='target_binario', 
            palette="viridis", ax=axes[1], edgecolor="black"
        )
        axes[1].set_title(f"Probabilidad de ser '{clase_favorable}' (100% Datos)", fontsize=13)
        axes[1].set_ylabel("Tasa de Éxito (0.0 a 1.0)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis='x', rotation=45)

        # Inyectar porcentajes sobre las barras
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.1%}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom', fontsize=10, color='black', 
                             xytext=(0, 4), textcoords='offset points')

        plt.tight_layout()
        plt.show()
        print("-" * 100)

    # Limpieza de basura temporal
    cols_a_limpiar = [c for c in df_viz.columns if c.endswith('_viz') or c == 'target_binario']
    if cols_a_limpiar: df_viz.drop(columns=cols_a_limpiar, inplace=True)

    print(f"⏱️ Renderizado categórico completado en {time.time() - inicio_timer:.2f}s")

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'y_train'.")

    # Extracción Automática de la clase minoritaria desde la memoria del Manager (rutas)
    clase_fav_dinamica = manager.rutas.get('clase_minoritaria', None)

    # Ejecutar el motor de visualización alimentándolo desde el manager
    radiografia_visual_categoricas(
        X=manager.X_train, 
        y=manager.y_train, 
        clase_favorable=clase_fav_dinamica, # Se usa la detectada en el Paso 4.1
        max_cats_visual=10      # Si una variable tiene 40 países, mostrará los 9 top y 1 "Otros"
    )

except Exception as e:
    print(f"🛑 Error en la visualización categórica: {e}")


# In[59]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

def radiografia_dispersion_cruzada(
    X: pd.DataFrame, 
    y: pd.Series, 
    max_cols=5, 
    max_rows_visual=2500
):
    """
    [FASE 2 - Paso 5.3] Matriz de Dispersión Cruzada AutoML (Pairplot).
    - Escudo Dimensional: Filtra las columnas con mayor varianza si hay demasiadas (Evita O(N^2) gráficos).
    - Escudo de RAM: Muestreo estratificado inteligente si el dataset es masivo (Evita colapso por overplotting).
    - Muestra cómo interactúan las numéricas en 2D coloreadas por el Target.
    """
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: Las matrices están vacías.")

    print(f"=== 🌌 FASE 5.3: Dispersión Cruzada Bivariada (Pairplot) ===")
    inicio_timer = time.time()
    warnings.simplefilter("ignore")
    sns.set_theme(style="white", context="notebook") # Fondo blanco para no saturar con mallas
    paleta_target = "Set2"

    # 1. Extracción y Ensamblaje Seguro
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print("  ⚠️ [INFO] No hay variables numéricas para cruzar.")
        return

    df_viz = X[num_cols].copy()
    target_name = y.name if y.name else 'Target'
    df_viz[target_name] = y

    # ==========================================
    # 🚀 ESCUDO DIMENSIONAL (Protección de Columnas)
    # ==========================================
    if len(num_cols) > max_cols:
        print(f"  ⚠️ [ALERTA DIMENSIONAL] {len(num_cols)} numéricas detectadas. El cruce generaría {len(num_cols)**2} gráficos.")
        print(f"  🛡️ Seleccionando el Top {max_cols} con mayor varianza para evitar caos visual...")
        # Ignoramos la varianza de los posibles IDs o ceros congelados, buscamos variables dinámicas
        varianzas = df_viz[num_cols].var().sort_values(ascending=False)
        mejores_cols = varianzas.head(max_cols).index.tolist()
        df_viz = df_viz[mejores_cols + [target_name]]
    else:
        mejores_cols = num_cols

    # ==========================================
    # 🚀 ESCUDO DE RAM (Protección de Filas)
    # ==========================================
    total_filas = len(df_viz)
    if total_filas > max_rows_visual:
        print(f"  ⚠️ [ALERTA DE RENDIMIENTO] Matriz masiva detectada ({total_filas:,} filas).")
        print(f"  ⚡ Protegiendo RAM: Muestreando {max_rows_visual:,} filas estratificadas para renderizado fluido...")

        # Fracción exacta para llegar a max_rows_visual
        fraccion = max_rows_visual / total_filas

        try:
            # Muestreo estratificado para no perder la proporción de la clase minoritaria
            df_viz = df_viz.groupby(target_name, group_keys=False).apply(lambda x: x.sample(frac=fraccion, random_state=42))
        except ValueError:
            # Fallback de seguridad: Si hay una clase extremadamente pequeña que rompe la fracción, usamos random simple
            df_viz = df_viz.sample(n=max_rows_visual, random_state=42)

    print(f"  📊 Generando Matriz de Interacción para: {mejores_cols}\n")

    # ==========================================
    # 4. Motor de Renderizado Pairplot
    # ==========================================
    # Usamos alpha para transparencia (overplotting) y s para el tamaño del punto
    g = sns.pairplot(
        df_viz, 
        hue=target_name, 
        palette=paleta_target, 
        diag_kind="kde",            # Montañas de densidad en la diagonal principal
        corner=True,                # Ocultar el triángulo superior (espejo redundante para ahorrar RAM)
        plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': None}
    )

    g.fig.suptitle(f"Matriz de Dispersión: ¿Cómo interactúan las variables para definir '{target_name}'?", 
                   y=1.02, fontsize=16, fontweight='bold')

    plt.show()
    print(f"⏱️ Matriz generada en {time.time() - inicio_timer:.2f}s")


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'y_train'.")

    # Ejecuta el escáner consumiendo los datos directamente del Manager.
    radiografia_dispersion_cruzada(
        X=manager.X_train, 
        y=manager.y_train, 
        max_cols=5, 
        max_rows_visual=2500
    )

except Exception as e:
    print(f"🛑 Error en el Pairplot: {e}")


# In[60]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
from IPython.display import display, Markdown

def cazar_colisiones_absolutas_seguro(X: pd.DataFrame, y: pd.Series):
    """
    [FASE 2 - Paso 6.1] Motor AutoML de Colisiones (Error de Bayes) - V2 Optimizada.
    - Busca filas donde TODAS las características (X) son idénticas, pero el Target es diferente.
    - PARCHE DE RAM: Utiliza lógica de conjuntos (drop_duplicates + merge) en lugar de un 
      groupby multidimensional para evitar la explosión de memoria (Producto Cartesiano de Pandas).
    - PARCHE PIPELINE: Devuelve la matriz original INTACTA para no perder datos.
    """
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("🛑 Error Crítico: Las matrices X_train o y_train están vacías.")

    print(f"=== 💥 FASE 6.1: Cacería de Colisiones Absolutas (Ruido Irreductible) ===")
    inicio_timer = time.time()

    # 1. Ensamblaje Seguro de la Matriz
    df_analisis = X.copy()
    target_name = y.name if y.name else 'Target'
    df_analisis[target_name] = y
    features = X.columns.tolist()
    total_filas = len(df_analisis)

    # ==========================================
    # 2. Lógica de Conjuntos (El Truco Anti-RAM)
    # ==========================================
    dups_x_mask = df_analisis.duplicated(subset=features, keep=False)
    df_sospechosos = df_analisis[dups_x_mask]

    if df_sospechosos.empty:
        print("  ✅ [MATRIZ PERFECTA] No hay clones de características. El Error de Bayes por colisión es 0%.")
        return X # Devolvemos la matriz intacta

    df_unicos_xy = df_sospechosos.drop_duplicates(subset=features + [target_name])

    colisiones_mask = df_unicos_xy.duplicated(subset=features, keep=False)
    df_colisiones_unicas = df_unicos_xy[colisiones_mask]

    if df_colisiones_unicas.empty:
        print("  ✅ [SIN CONTRADICCIONES] Hay filas duplicadas, pero todas coinciden en su Target. No hay colisiones absolutas.")
        return X # Devolvemos la matriz intacta

    claves_colision = df_colisiones_unicas[features].drop_duplicates()
    df_colisiones_finales = pd.merge(df_analisis, claves_colision, on=features, how='inner')

    # ==========================================
    # 3. Cálculo de Impacto
    # ==========================================
    filas_afectadas = len(df_colisiones_finales)
    porcentaje_ruido = (filas_afectadas / total_filas) * 100

    # ==========================================
    # 4. Reporte Ejecutivo y Diagnóstico Seguro
    # ==========================================
    display(Markdown(f"### 🚨 Alerta de Contradicción: **{filas_afectadas} filas** afectadas ({porcentaje_ruido:.2f}% del dataset)."))
    print(f"  🧠 Significado: Estas {filas_afectadas} filas forman perfiles idénticos pero con ingresos opuestos.")
    print(f"  📉 Límite Teórico: Debido a este ruido, tu modelo NUNCA podrá alcanzar el 100% de precisión.\n")

    print("  🔍 TOP 5 Perfiles con mayor nivel de colisión:")

    df_colisiones_finales['Perfil_ID (Firma)'] = df_colisiones_finales[features].astype(str).agg(' | '.join, axis=1)

    resumen = df_colisiones_finales.groupby('Perfil_ID (Firma)')[target_name].value_counts().unstack(fill_value=0)
    resumen['Total_Clones'] = resumen.sum(axis=1)
    resumen = resumen.sort_values(by='Total_Clones', ascending=False).head(5)

    display(resumen)

    print("-" * 100)
    print("  💡 ACCIÓN SUGERIDA (Fase 3):")
    print("  En datasets tabulares, solemos DEJAR ESTAS FILAS INTACTAS. Los algoritmos como XGBoost ")
    print("  usarán la probabilidad (ej. si hay 8 pobres y 2 ricos en el grupo, predecirá 'pobre' con 80% de certeza).")

    print(f"\n⏱️ Escáner de colisiones completado en {time.time() - inicio_timer:.3f}s")

    # ==========================================
    # 🚀 CORRECCIÓN CRÍTICA: Devolver la matriz completa
    # ==========================================
    return X 

# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X' not in locals() or 'y' not in locals():
        raise EnvironmentError("No se encontraron las variables 'X' o 'y'. Ejecuta el aislamiento (Paso 3.2) primero.")

    # Ejecutar el análisis usando las matrices completas originales.
    # NO guardamos el resultado en 'X' para evitar desincronizaciones de filas.
    # El motor solo escanea y reporta.
    _ = cazar_colisiones_absolutas_seguro(X=X, y=y)

    print("\n✅ Diagnóstico finalizado. Las matrices X e y siguen intactas y balanceadas.")

except Exception as e:
    print(f"🛑 Error en el escáner de colisiones: {e}")


# In[61]:


X.info()


# In[62]:


y.info()


# In[63]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
import re

def aplicar_rare_labeling_seguro(
    X: pd.DataFrame, 
    umbral: float = 0.01, 
    etiqueta_rara: str = 'Rare'
):
    """
    [FASE 2 - Paso 7.1] Motor AutoML de Rare Labeling (Agrupación de Colas Largas).
    - Escanea columnas de texto/categóricas y agrupa categorías minoritarias (< umbral).
    - MLOPS SHIELD (Regex): Auto-detecta y protege explícitamente los NaNs camuflados 
      (textos nulos o símbolos puros) para no destruirlos antes de la Fase 3.
    - Retorna el DataFrame transformado y un 'diccionario de estado' para aplicar a X_test después.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🛡️ FASE 7.1: Escudo de Alta Cardinalidad (Rare Labeling al {umbral*100}%) ===")
    inicio_timer = time.time()

    # Operamos sobre una copia para no alterar accidentalmente memorias globales
    X_transformado = X.copy()

    # 1. Aislar solo las columnas que son texto o categorías
    cat_cols = X_transformado.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    if not cat_cols:
        print("  ⚠️ [INFO] No se detectaron variables categóricas para agrupar.")
        return X_transformado, {}

    diccionario_vocabulario = {}
    columnas_modificadas = 0

    # MOTOR REGEX: Atrapa nulos comunes, -1 como string, o cadenas puras de símbolos ASCII
    patron_mascara = re.compile(r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|)$|^[^a-zA-Z0-9]+$')

    print(f"  🔍 Escaneando {len(cat_cols)} variables categóricas...\n")

    # ==========================================
    # 2. Motor de Evaluación y Colapso
    # ==========================================
    for col in cat_cols:
        # Calculamos la frecuencia relativa (ignorando NaNs reales temporalmente)
        frecuencias = X_transformado[col].value_counts(normalize=True)

        # A. Identificar el "Club VIP" (Categorías que superan el 1%)
        categorias_validas = frecuencias[frecuencias >= umbral].index.tolist()

        # B. ESCUDO MLOPS INTELIGENTE: Usamos la Regex para cazar máscaras en esta columna
        # Casteamos val a str() por si acaso atrapa un -1 numérico o un bool
        mascaras_presentes = [val for val in frecuencias.index if patron_mascara.match(str(val).strip())]

        # Inyectamos las máscaras al Club VIP para que el Rare Labeling no las toque
        categorias_validas.extend(mascaras_presentes)
        categorias_validas = list(set(categorias_validas)) # Eliminar duplicados si los hubiera

        # C. Identificar a los que van a la guillotina (Los que NO están en el Club VIP)
        categorias_raras = frecuencias[~frecuencias.index.isin(categorias_validas)].index.tolist()

        if categorias_raras:
            columnas_modificadas += 1
            # Guardamos el vocabulario oficial. Esto es VITAL para aplicar el .transform() a X_test
            diccionario_vocabulario[col] = categorias_validas

            # Construimos una máscara booleana segura para reemplazar
            # IMPORTANTE: Protegemos los np.nan reales para no sobreescribirlos con texto 'Rare'
            mascara_nulos = X_transformado[col].isna()
            mascara_reemplazo = ~X_transformado[col].isin(categorias_validas) & ~mascara_nulos

            cantidad_reemplazada = mascara_reemplazo.sum()
            porcentaje_reemplazado = (cantidad_reemplazada / len(X_transformado)) * 100

            # --- SOLUCIÓN DEL ERROR TypeError ---
            # Si la columna es de tipo 'category', debemos añadir explícitamente la
            # nueva categoría 'Rare' antes de intentar asignar valores a ella.
            if pd.api.types.is_categorical_dtype(X_transformado[col]):
                X_transformado[col] = X_transformado[col].cat.add_categories([etiqueta_rara])
            # ------------------------------------

            # Aplicar la agrupación (Aquí es donde fallaba antes)
            X_transformado.loc[mascara_reemplazo, col] = etiqueta_rara

            # Generar reporte dinámico
            if col == 'native_country':
                print(f"  🌎 '{col}': {len(categorias_raras)} países minoritarios colapsados en '{etiqueta_rara}' ({porcentaje_reemplazado:.1f}% del volumen).")
            else:
                print(f"  🔧 '{col}': {len(categorias_raras)} categorías colapsadas a '{etiqueta_rara}' ({porcentaje_reemplazado:.1f}% del volumen).")

            if mascaras_presentes:
                print(f"      ↳ 🛡️ Máscaras protegidas del colapso: {mascaras_presentes}")

    print("-" * 80)
    if columnas_modificadas == 0:
        print("  ✅ [MATRIZ ROBUSTA] Ninguna categoría cayó por debajo del umbral. No se requirió agrupación.")
    else:
        print(f"  📦 El diccionario 'fit' de vocabulario ha sido generado exitosamente para {columnas_modificadas} columnas.")

    print(f"\n⏱️ Rare Labeling completado en {time.time() - inicio_timer:.3f}s")

    # Retornamos la matriz purgada y el diccionario para X_test
    return X_transformado, diccionario_vocabulario

# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X' not in locals() and 'X' not in globals():
        raise EnvironmentError("No se encontró 'X'.")

    # Ejecutamos el escudo sobre nuestra matriz unificada (FASE 2)
    # El umbral 0.01 significa 1%. Todo lo que tenga menos del 1% se vuelve 'Rare'
    X, diccionario_rare_labels = aplicar_rare_labeling_seguro(
        X=X, 
        umbral=0.01, 
        etiqueta_rara='Rare'
    )

    print("\n✅ Matriz X actualizada globalmente. Lista para continuar hacia la Fase 3.")

except Exception as e:
    print(f"🛑 Error en el Escudo de Cardinalidad: {e}")


# In[64]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time

# Pydantic se usa en MLOps para forzar que el LLM devuelva la estructura exacta que necesitamos
try:
    from pydantic import BaseModel, Field
    PYDANTIC_DISPONIBLE = True
except ImportError:
    PYDANTIC_DISPONIBLE = False

# 1. Definimos el Esquema Estricto que le exigiremos al LLM
if PYDANTIC_DISPONIBLE:
    class ExtraccionLLM(BaseModel):
        sentimiento: str = Field(description="Clasificar como: Positivo, Negativo o Neutral")
        entidad_clave: str = Field(description="La palabra o concepto principal del texto")
        alerta_riesgo: bool = Field(description="True si el texto indica peligro, fraude o riesgo alto, False de lo contrario")

def fusion_semantica_llm(X: pd.DataFrame, y: pd.Series = None):
    """
    [FASE 2 - Paso 7.2] Motor AutoML de Fusión Semántica para Texto Libre.
    - Radar Inteligente: Detecta columnas que realmente son texto libre (alta longitud y cardinalidad).
    - MLOps Pipeline: Maqueta la extracción estructurada (JSON) usando un LLM ligero.
    - Bypass Automático: Si no hay texto libre (ej. Dataset Adult), se omite sin romper el flujo.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🧠 FASE 7.2: Fusión Semántica con LLMs (Extracción Estructurada) ===")
    inicio_timer = time.time()
    X_transformado = X.copy()

    # ==========================================
    # 2. Radar Inteligente de Texto Libre
    # ==========================================
    text_cols = X_transformado.select_dtypes(include=['object', 'string']).columns.tolist()
    columnas_texto_libre = []

    for col in text_cols:
        s = X_transformado[col].dropna().astype(str)
        if s.empty: continue

        # Heurísticas de Texto Libre: 
        # 1. Longitud promedio mayor a 35 caracteres (una categoría normal mide menos)
        # 2. Alta cardinalidad: Al menos el 50% de las filas tienen un texto distinto
        longitud_promedio = s.str.len().mean()
        ratio_unicos = s.nunique() / len(s)

        if longitud_promedio > 35 and ratio_unicos > 0.5:
            columnas_texto_libre.append(col)

    if not columnas_texto_libre:
        print("  ✅ [BYPASS AUTOMÁTICO] No se detectaron columnas de Texto Libre (Free Text).")
        print("  El dataset contiene solo categorías estructuradas. Omitiendo inferencia LLM.")
        print(f"\n⏱️ Análisis de texto omitido en {time.time() - inicio_timer:.3f}s")
        return X_transformado

    print(f"  📖 [TEXTO DETECTADO] Variables de texto libre a procesar: {columnas_texto_libre}")
    if not PYDANTIC_DISPONIBLE:
        print("  ⚠️ Advertencia: Pydantic no está instalado. Instálalo para garantizar la estructura del JSON.")

    # ==========================================
    # 3. Motor de Inferencia LLM (Arquitectura Mockup para Producción)
    # ==========================================
    def invocar_llm_local(texto: str) -> dict:
        """
        Aquí iría la llamada a tu LLM local (ej. Llama.cpp, Ollama, vLLM) 
        o a una API si está permitido. Para el template, simulamos la respuesta.
        """
        # --- SIMULACIÓN PARA QUE EL CÓDIGO CORRA ---
        # Si el texto estuviera vacío o fuera un NaN
        if pd.isna(texto) or str(texto).strip() in ['?', '']:
            return {"sentimiento": "Neutral", "entidad_clave": "Ninguna", "alerta_riesgo": False}

        return {"sentimiento": "Neutral", "entidad_clave": "Concepto_Genérico", "alerta_riesgo": False}

    # ==========================================
    # 4. Procesamiento por Lotes (Batch Processing)
    # ==========================================
    for col in columnas_texto_libre:
        print(f"  🤖 Extrayendo semántica de '{col}'...")

        # Extraemos las respuestas (simuladas) en una lista de diccionarios
        respuestas_estructuradas = X_transformado[col].apply(invocar_llm_local)

        # Expandimos el JSON en columnas nativas de Pandas
        df_extraido = pd.json_normalize(respuestas_estructuradas)
        df_extraido.columns = [f"{col}_LLM_{c}" for c in df_extraido.columns]

        # Concatenamos las nuevas características a la matriz y eliminamos el texto crudo original
        X_transformado = pd.concat([X_transformado.reset_index(drop=True), df_extraido.reset_index(drop=True)], axis=1)
        X_transformado.drop(columns=[col], inplace=True)

        print(f"    ↳ Creadas {len(df_extraido.columns)} nuevas columnas estructuradas.")

    print(f"\n⏱️ Fusión Semántica completada en {time.time() - inicio_timer:.3f}s")

    return X_transformado

# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X' not in locals() and 'X' not in globals():
        raise EnvironmentError("No se encontró 'X'.")

    # Ejecutamos el motor semántico sobre la matriz global (Fase 2)
    # Hará bypass seguro en el dataset Adult porque no tiene columnas de comentarios largos
    X = fusion_semantica_llm(X=X)

    print("\n✅ Matriz X lista.")

except Exception as e:
    print(f"🛑 Error en la Fusión Semántica: {e}")


# # FASE 3: Ingeniería Básica y Codificación (El Puente Matemático)
# Extraemos métricas directas y convertimos TODO a números para que los imputadores funcionen.

# In[65]:


X.head()


# In[66]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import re

def desenmascarar_falsos_nulos(
    X: pd.DataFrame, 
    nulos_numericos_conocidos: dict = None
):
    """
    [FASE 3 - Paso 8.1] Motor AutoML para Desenmascarar Falsos Nulos.
    - Caza nulos categóricos usando una Regex de símbolos ASCII y palabras clave comunes.
    - Caza nulos numéricos (Outliers lógicos) inyectados vía diccionario (ej. 99999.0).
    - Caza Outliers Temporales (NUEVO): Detecta fechas imposibles (ej. 1900 o 2099) y las vuelve NaT.
    - Convierte todo el ruido encontrado estandarizadamente a np.nan/pd.NaT y reporta los hallazgos.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🧹 FASE 8.1: Desenmascaramiento de Falsos Nulos (Missingness) ===")
    inicio_timer = time.time()

    # Operamos sobre una copia limpia
    X_transformado = X.copy()
    total_filas = len(X_transformado)

    # MOTOR REGEX: 
    # 1. (unknown|n/?a|null|nan|missing|none|-1|) -> Palabras clave ignorando mayúsculas/minúsculas.
    # 2. ^[^a-zA-Z0-9]+$ -> Cualquier cadena que NO contenga letras ni números (ej. "?", "-", "**", "  ").
    patron_mascara = re.compile(r'(?i)^(unknown|n/?a|null|nan|missing|none|-1|nat|)$|^[^a-zA-Z0-9]+$')

    celdas_desenmascaradas_totales = 0

    # ==========================================
    # 1. Purga de Columnas Categóricas / Texto
    # ==========================================
    cat_cols = X_transformado.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    print(f"  🔍 Escaneando {len(cat_cols)} variables de texto con Motor Regex...")

    for col in cat_cols:
        evaluacion_regex = X_transformado[col].astype(str).str.strip().str.match(patron_mascara)
        mascara_falsos_nulos = evaluacion_regex & X_transformado[col].notna()

        hallazgos = mascara_falsos_nulos.sum()
        if hallazgos > 0:
            valores_encontrados = X_transformado.loc[mascara_falsos_nulos, col].unique().tolist()
            porcentaje_columna = (hallazgos / total_filas) * 100

            X_transformado.loc[mascara_falsos_nulos, col] = np.nan
            celdas_desenmascaradas_totales += hallazgos
            print(f"    🎭 '{col}': {hallazgos} máscaras {valores_encontrados} destruidas y convertidas a NaN ({porcentaje_columna:.2f}%).")

    # ==========================================
    # 2. Purga de Columnas Numéricas (Trampas Lógicas)
    # ==========================================
    if nulos_numericos_conocidos:
        print(f"\n  🔢 Escaneando variables numéricas en busca de trampas conocidas...")
        for col, valores_trampa in nulos_numericos_conocidos.items():
            if col in X_transformado.columns:
                mascara_numerica = X_transformado[col].isin(valores_trampa)
                hallazgos_num = mascara_numerica.sum()

                if hallazgos_num > 0:
                    valores_encontrados_num = X_transformado.loc[mascara_numerica, col].unique().tolist()
                    porcentaje_col_num = (hallazgos_num / total_filas) * 100

                    X_transformado.loc[mascara_numerica, col] = np.nan
                    celdas_desenmascaradas_totales += hallazgos_num
                    print(f"    💣 '{col}': {hallazgos_num} valores trampa {valores_encontrados_num} convertidos a NaN ({porcentaje_col_num:.2f}%).")

    # ==========================================
    # 🚀 2.5 Purga de Fechas Ilógicas (Outliers Temporales)
    # ==========================================
    date_cols = X_transformado.select_dtypes(include=['datetime64', 'datetimetz', 'datetime']).columns.tolist()

    if date_cols:
        print(f"\n  ⏳ Escaneando {len(date_cols)} variables temporales en busca de fechas ilógicas...")

        anio_actual = pd.Timestamp.now().year
        umbral_pasado = 1900 # Fechas anteriores a 1900 suelen ser errores/placeholders
        umbral_futuro = anio_actual + 2 # Más de 2 años en el futuro suele ser un typo

        for col in date_cols:
            anios_columna = X_transformado[col].dt.year

            # Máscara inteligente ignorando NaNs preexistentes
            mask_pasado = anios_columna < umbral_pasado
            mask_futuro = anios_columna > umbral_futuro
            mask_outliers_fechas = mask_pasado | mask_futuro

            hallazgos_fechas = mask_outliers_fechas.sum()

            if hallazgos_fechas > 0:
                # Extraemos muestra para el log
                valores_fechas = X_transformado.loc[mask_outliers_fechas, col].dt.strftime('%Y-%m-%d').unique().tolist()
                valores_muestra = valores_fechas[:3] + ["..."] if len(valores_fechas) > 3 else valores_fechas
                porcentaje_fechas = (hallazgos_fechas / total_filas) * 100

                # Conversión estricta a Not a Time (NaT)
                X_transformado.loc[mask_outliers_fechas, col] = pd.NaT
                celdas_desenmascaradas_totales += hallazgos_fechas
                print(f"    🕰️ '{col}': {hallazgos_fechas} fechas imposibles {valores_muestra} convertidas a NaT ({porcentaje_fechas:.2f}%).")

    # ==========================================
    # 3. Reporte de Impacto
    # ==========================================
    print("-" * 80)
    if celdas_desenmascaradas_totales > 0:
        porcentaje_total = (celdas_desenmascaradas_totales / (X_transformado.shape[0] * X_transformado.shape[1])) * 100
        print(f"  ✅ [PURGA EXITOSA] Se desenmascararon {celdas_desenmascaradas_totales} celdas falsas ({porcentaje_total:.2f}% de la matriz total).")
        print("  🧠 El modelo ahora sabe exactamente dónde hay agujeros de información reales.")
    else:
        print("  ✅ [MATRIZ LIMPIA] No se detectaron máscaras, valores trampa, ni fechas ilógicas.")

    print(f"\n⏱️ Desenmascaramiento completado en {time.time() - inicio_timer:.3f}s")

    return X_transformado

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    print(">>> Aplicando Desenmascaramiento a TRAIN <<<")
    X_train_purgado = desenmascarar_falsos_nulos(
        X=manager.X_train,
        nulos_numericos_conocidos={'capital_gain': [99999, 99999.0]}
    )

    print("\n>>> Aplicando Desenmascaramiento a TEST <<<")
    X_test_purgado = desenmascarar_falsos_nulos(
        X=manager.X_test,
        nulos_numericos_conocidos={'capital_gain': [99999, 99999.0]}
    )

    # Guardar en la memoria del Manager
    manager.X_train = X_train_purgado
    manager.X_test = X_test_purgado

    print("\n✅ Matrices purgadas en el Manager. Todos los nulos/fechas falsas son ahora np.nan o pd.NaT.")

    # (Transición) Reflejar temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test

except Exception as e:
    print(f"🛑 Error en el desenmascaramiento: {e}")


# In[67]:


X_train.info()


# In[68]:


X_test.info()


# In[69]:


X_train.head()


# In[70]:


X_test[10:20]


# In[71]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time

def aplicar_missingness_flags(
    X: pd.DataFrame, 
    prefijo: str = 'is_missing_',
    columnas_aprendidas: list = None
) -> tuple:
    """
    [FASE 3 - Paso 8.2] Motor AutoML de Rastreo de Nulos (Missingness).
    - Muro de Hierro MLOps: Aprende las columnas con nulos en Train, y las replica exactamente en Test.
    1. Banderas Booleanas: Crea columnas indicadoras (1/0) para variables con nulos.
    2. Row-wise NaN Count: Inyecta una característica maestra con el total de nulos por individuo.
    - MLOPS SHIELD: Usa np.int8 y retorna las listas para actualizar el enrutamiento dinámicamente.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🚩 FASE 8.2: Rastreadores de Nulidad (Missingness Flags & Row-wise Count) ===")
    inicio_timer = time.time()

    X_transformado = X.copy()
    banderas_creadas = []

    # ==========================================
    # 1. MLOps: Fit vs Transform (Alineación de Matrices)
    # ==========================================
    if columnas_aprendidas is None:
        # MODO TRAIN (.fit): Detectamos nosotros mismos dónde hay nulos
        columnas_con_nulos = X_transformado.columns[X_transformado.isna().any()].tolist()
    else:
        # MODO TEST (.transform): Usamos estrictamente lo que nos dictó Train
        columnas_con_nulos = columnas_aprendidas

    if not columnas_con_nulos:
        print("  ✅ [MATRIZ PERFECTA] No hay nulos detectados. Se omite la creación de banderas.")
        return X_transformado, [], None, None # FIX: Agregado None extra para que el unpack no falle si no hay nulos

    if columnas_aprendidas is None:
        print(f"  🔍 Detectadas {len(columnas_con_nulos)} variables con agujeros de información (Modo Aprendizaje).")
    else:
        print(f"  🔒 Replicando {len(columnas_con_nulos)} rastreadores aprendidos de Train (Modo Aplicación).")

    print("  ⚙️ Generando rastreadores...\n")

    # ==========================================
    # 2. Fabricación de Banderas Booleanas por Columna
    # ==========================================
    for col in columnas_con_nulos:
        nombre_bandera = f"{prefijo}{col}"
        # astype(np.int8) convierte True/False en 1/0 pesando solo 1 byte por fila
        X_transformado[nombre_bandera] = X_transformado[col].isna().astype(np.int8)
        banderas_creadas.append(nombre_bandera)
        print(f"    ↳ Creada bandera booleana: '{nombre_bandera}'")

    # ==========================================
    # 3. Fabricación del "Row-wise NaN Count" (La Variable Maestra)
    # ==========================================
    nombre_conteo = 'total_nulos_en_fila'

    # Calculamos cuántos nulos hay en la matriz original por cada persona
    conteo_nulos_por_fila = X_transformado[columnas_con_nulos].isna().sum(axis=1)

    # Inyectamos el conteo en int8
    X_transformado[nombre_conteo] = conteo_nulos_por_fila.astype(np.int8)

    # ==========================================
    # 4. Reporte Ejecutivo
    # ==========================================
    max_nulos = conteo_nulos_por_fila.max()
    filas_afectadas = (conteo_nulos_por_fila > 0).sum()
    porcentaje_filas = (filas_afectadas / len(X_transformado)) * 100

    print("-" * 80)
    print(f"  📊 Característica Maestra '{nombre_conteo}' inyectada con éxito.")
    print(f"  👥 Impacto: {filas_afectadas} individuos ({porcentaje_filas:.2f}%) ocultaron al menos 1 dato.")
    print(f"  ⚠️ El récord máximo de datos faltantes en una sola persona es: {max_nulos} nulos.")

    print(f"\n⏱️ Motor de Missingness completado en {time.time() - inicio_timer:.3f}s")

    # Retornamos las variables nuevas (y la lista de columnas base para pasársela a Test)
    return X_transformado, banderas_creadas, nombre_conteo, columnas_con_nulos

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    print("\n>>> 🚂 ENTRENANDO RASTREADORES EN TRAIN <<<")
    # En Train no le pasamos 'columnas_aprendidas' para que las descubra por sí mismo
    X_train_miss, flags_nulidad, var_conteo, cols_aprendidas_train = aplicar_missingness_flags(
        X=manager.X_train
    )

    print("\n>>> 🔒 APLICANDO RASTREADORES A TEST <<<")
    # En Test le forzamos la lista exacta de columnas que descubrimos en Train
    X_test_miss, _, _, _ = aplicar_missingness_flags(
        X=manager.X_test,
        columnas_aprendidas=cols_aprendidas_train
    )

    # Guardamos los resultados de vuelta en el Manager
    manager.X_train = X_train_miss
    manager.X_test = X_test_miss

    # MLOPS TIP: Actualizamos nuestra lista de rutas dinámicamente en el Manager
    if flags_nulidad:
        # Usamos list comprehension para agregar solo los flags que no estén ya en la ruta
        nuevos_bools = [f for f in flags_nulidad if f not in manager.rutas['bool_vars']]
        manager.rutas['bool_vars'].extend(nuevos_bools)

        if var_conteo and var_conteo not in manager.rutas['num_vars']:
            manager.rutas['num_vars'].append(var_conteo)

        print(f"\n🛣️ Ruteo AutoML actualizado en Manager: +{len(nuevos_bools)} bools, +1 nums.")

    # (Transición) Reflejar temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en el motor de missingness: {e}")


# In[72]:


bool_vars


# In[73]:


X_train.info()


# In[74]:


X_test.info()


# In[75]:


# ==========================================
# 0. Blindaje de Dependencias
# ==========================================
import pandas as pd
import numpy as np
import time

def ingenieria_banderas_negocio(
    X: pd.DataFrame, 
    umbral_ceros: float = 0.85,
    rutas_actuales: dict = None,
    reglas_aprendidas: dict = None
) -> tuple:
    """
    [FASE 3 - Paso 8.3] Motor AutoML de Ingeniería de Características (Auto-Descubrimiento).
    - Escáner de Dispersión: Detecta y escuda variables numéricas con exceso de ceros.
    - MLOPS SHIELD: Ignora inteligentemente banderas previas para evitar recursividad.
    - AUTO-FEATURE CROSSES: Detecta automáticamente pares lógicos y genera variables netas.
    - Alineación Train/Test: Aprende las reglas en Train y las fuerza ciegamente en Test.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz predictora (X) está vacía.")

    # ---------------------------------------------------------
    # SHIELD: Construimos la lista negra leyendo el historial
    # ---------------------------------------------------------
    columnas_ignoradas = []
    if rutas_actuales and 'bool_vars' in rutas_actuales:
        columnas_ignoradas.extend(rutas_actuales['bool_vars'])
        if 'total_nulos_en_fila' in X.columns:
            columnas_ignoradas.append('total_nulos_en_fila')

    print(f"=== 🚀 FASE 8.3: Ingeniería de Características (Escudos Automáticos y Atajos) ===")
    inicio_timer = time.time()

    X_transformado = X.copy()
    nuevas_bools = []
    nuevas_nums = []
    reglas_actuales = {'columnas_bandera': [], 'parejas_cruce': []} if reglas_aprendidas is None else reglas_aprendidas

    # ==========================================
    # 1. Escáner Inteligente de Banderas (Auto-Sparsity Flags)
    # ==========================================
    cols_numericas = X_transformado.select_dtypes(include=['number']).columns.tolist()

    if reglas_aprendidas is None:
        print(f"  ⚙️ [TRAIN] Escaneando variables numéricas con más de {umbral_ceros*100}% de ceros...")
        for col in cols_numericas:
            if col in columnas_ignoradas:
                continue

            valores_unicos = set(X_transformado[col].dropna().unique())
            if valores_unicos.issubset({0, 1, 0.0, 1.0}):
                continue

            total_validos = X_transformado[col].notna().sum()
            if total_validos == 0: continue

            ratio_ceros = (X_transformado[col] == 0).sum() / total_validos

            if ratio_ceros >= umbral_ceros:
                reglas_actuales['columnas_bandera'].append(col)
                nombre_bandera = f"tiene_{col}"
                X_transformado[nombre_bandera] = (X_transformado[col].fillna(0) != 0).astype(np.int8)
                nuevas_bools.append(nombre_bandera)
                print(f"    🌟 [Atajo AutoML] '{col}' ({ratio_ceros*100:.1f}% ceros) -> Creada bandera: '{nombre_bandera}'")
    else:
        print(f"  🔒 [TEST] Aplicando {len(reglas_actuales['columnas_bandera'])} banderas aprendidas de Train...")
        for col in reglas_actuales['columnas_bandera']:
            if col in X_transformado.columns:
                nombre_bandera = f"tiene_{col}"
                X_transformado[nombre_bandera] = (X_transformado[col].fillna(0) != 0).astype(np.int8)
                nuevas_bools.append(nombre_bandera)
                print(f"    ↳ Replicada bandera: '{nombre_bandera}'")

    # ==========================================
    # 2. Auto-Descubrimiento de Interacciones (Feature Crosses)
    # ==========================================
    patrones_opuestos = [
        ('_gain', '_loss'), 
        ('_ingreso', '_gasto'),
        ('_max', '_min'),
        ('positive_', 'negative_')
    ]

    if reglas_aprendidas is None:
        print("\n  ⚙️ [TRAIN] Escaneando matriz en busca de cruces matemáticos lógicos (Auto-Discovery)...")
        for sufijo_a, sufijo_b in patrones_opuestos:
            cols_a = [c for c in cols_numericas if c.endswith(sufijo_a) or c.startswith(sufijo_a)]
            for col_a in cols_a:
                base_name = col_a.replace(sufijo_a, "")
                col_b = base_name + sufijo_b if col_a.endswith(sufijo_a) else sufijo_b + base_name

                if col_b in cols_numericas:
                    nuevo_nombre = f"{base_name}_neto" if col_a.endswith(sufijo_a) else f"neto_{base_name}"
                    reglas_actuales['parejas_cruce'].append((col_a, col_b, nuevo_nombre))

                    if nuevo_nombre not in X_transformado.columns:
                        X_transformado[nuevo_nombre] = X_transformado[col_a].fillna(0) - X_transformado[col_b].fillna(0)
                        nuevas_nums.append(nuevo_nombre)
                        print(f"    ⚖️ [Auto-Cruce Exitoso] Creada variable '{nuevo_nombre}' ({col_a} - {col_b})")
    else:
        print(f"\n  🔒 [TEST] Aplicando {len(reglas_actuales['parejas_cruce'])} cruces aprendidos de Train...")
        for col_a, col_b, nuevo_nombre in reglas_actuales['parejas_cruce']:
            if col_a in X_transformado.columns and col_b in X_transformado.columns:
                 X_transformado[nuevo_nombre] = X_transformado[col_a].fillna(0) - X_transformado[col_b].fillna(0)
                 nuevas_nums.append(nuevo_nombre)
                 print(f"    ↳ Replicado cruce: '{nuevo_nombre}'")

    # ==========================================
    # 3. Reporte de Impacto
    # ==========================================
    total_nuevas = len(nuevas_bools) + len(nuevas_nums)
    print("-" * 80)
    if total_nuevas > 0:
        print(f"  ✅ [INGENIERÍA EXITOSA] Se inyectaron {total_nuevas} variables dinámicas al modelo.")
    else:
        print("  ⚠️ [MATRIZ DENSA] No se detectó alta dispersión ni se aplicaron cruces.")

    print(f"\n⏱️ Ingeniería completada en {time.time() - inicio_timer:.3f}s")

    return X_transformado, nuevas_bools, nuevas_nums, reglas_actuales

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 ENTRENANDO INGENIERÍA EN TRAIN <<<")
    X_train_eng, flags_creadas, cruces_creados, reglas_ingenieria = ingenieria_banderas_negocio(
        X=manager.X_train, 
        umbral_ceros=0.85, 
        rutas_actuales=manager.rutas
    )

    print("\n>>> 🔒 APLICANDO INGENIERÍA A TEST <<<")
    X_test_eng, _, _, _ = ingenieria_banderas_negocio(
        X=manager.X_test, 
        umbral_ceros=0.85, 
        rutas_actuales=manager.rutas,
        reglas_aprendidas=reglas_ingenieria
    )

    # Guardamos en la memoria del Manager
    manager.X_train = X_train_eng
    manager.X_test = X_test_eng

    # MLOPS TIP: Actualizamos el ruteo en el Manager SOLO UNA VEZ con los descubrimientos de Train
    if flags_creadas or cruces_creados:
        manager.rutas['bool_vars'].extend(flags_creadas)
        manager.rutas['num_vars'].extend(cruces_creados)
        print(f"\n🛣️ Ruteo AutoML actualizado en Manager: +{len(flags_creadas)} bools, +{len(cruces_creados)} nums.")

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la ingeniería de características: {e}")


# In[76]:


X_train.info()


# In[77]:


X_test.info()


# In[78]:


y_train.info()


# In[79]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict

def diagnostico_topologico_automl(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    [FASE 9 - Paso 9.1] Escáner de Diagnóstico de Topología de Datos (MLOps Estricto).
    - FIX MLOps: La Caja Fuerte. Si detecta una entidad, la envía al Index 
      para blindarla contra Fases futuras (Encoders/KNN) asegurando su supervivencia hasta la 14.1.
    - DUPLICACIÓN INTELIGENTE (NUEVO): Usa drop=False para copiar al index pero dejar la columna viva.
    """
    if not isinstance(X_train, pd.DataFrame) or X_train.empty:
        raise ValueError("🛑 Error Crítico: La matriz X_train está vacía o es inválida.")

    print("=== 🔬 FASE 9.1: Diagnóstico de Topología del Dataset (AutoML) ===")
    inicio_timer = time.time()

    X_tr_analisis = X_train.copy()
    X_te_analisis = X_test.copy() if X_test is not None else None

    # 1. Liberación temporal de IDs del Index para poder analizarlos
    nombres_originales = [n for n in X_tr_analisis.index.names if n is not None]
    if nombres_originales:
        print(f"  🔓 Liberando temporalmente IDs del Index para diagnóstico: {nombres_originales}")
        X_tr_analisis = X_tr_analisis.reset_index()
        if X_te_analisis is not None:
            X_te_analisis = X_te_analisis.reset_index()

    reporte = {
        'topologia': 'Transversal',
        'columna_tiempo': None,
        'columna_entidad': None,
        'es_serie_tiempo_estricta': False
    }

    # ==========================================
    # Búsqueda de Relojes y Varianza
    # ==========================================
    cols_tiempo = X_tr_analisis.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()

    if not cols_tiempo:
        print("  ✅ [DIAGNÓSTICO] No se encontraron ejes temporales (Datetimes).")
        print("    ↳ Topología deducida: TRANSVERSAL (Cross-Sectional).")
        if nombres_originales: # Restaurar index original si existía
            # 🚀 FIX: Mantenemos las columnas originales vivas
            X_tr_analisis = X_tr_analisis.set_index(nombres_originales, drop=False)
            if X_te_analisis is not None: X_te_analisis = X_te_analisis.set_index(nombres_originales, drop=False)
        return X_tr_analisis, X_te_analisis, reporte

    col_tiempo = X_tr_analisis[cols_tiempo].nunique().idxmax() if len(cols_tiempo) > 1 else cols_tiempo[0]
    reporte['columna_tiempo'] = col_tiempo

    total_filas = len(X_tr_analisis)
    filas_validas = total_filas - X_tr_analisis[col_tiempo].isna().sum()
    unicos_tiempo = X_tr_analisis[col_tiempo].nunique()

    if unicos_tiempo <= 1:
        print(f"  ✅ [DIAGNÓSTICO] El Reloj '{col_tiempo}' está congelado en Train (Varianza Cero).")
        print("    ↳ Topología deducida: TRANSVERSAL (Cross-Sectional Snapshot).")
        if nombres_originales:
            X_tr_analisis = X_tr_analisis.set_index(nombres_originales, drop=False)
            if X_te_analisis is not None: X_te_analisis = X_te_analisis.set_index(nombres_originales, drop=False)
        return X_tr_analisis, X_te_analisis, reporte

    ratio_unicidad = unicos_tiempo / filas_validas if filas_validas > 0 else 0

    if ratio_unicidad >= 0.95:
        print(f"  ✅ [DIAGNÓSTICO] El tiempo fluye perfectamente (Unicidad: {ratio_unicidad:.1%}).")
        print(f"    ↳ Topología deducida: SERIE DE TIEMPO PURA.")
        reporte['topologia'] = 'Serie de Tiempo Pura'
        reporte['es_serie_tiempo_estricta'] = True
        if nombres_originales:
            X_tr_analisis = X_tr_analisis.set_index(nombres_originales, drop=False)
            if X_te_analisis is not None: X_te_analisis = X_te_analisis.set_index(nombres_originales, drop=False)
        return X_tr_analisis, X_te_analisis, reporte

    # ==========================================
    # Búsqueda de Entidades (Datos de Panel)
    # ==========================================
    print(f"  ⚠️ [ANÁLISIS PROFUNDO] Detectados múltiples eventos en la misma marca de tiempo (Unicidad: {ratio_unicidad:.1%}).")
    print("    ↳ Buscando una variable Categórica/ID que actúe como Llave Separadora (Entidad)...")

    posibles_entidades = X_tr_analisis.select_dtypes(include=['category', 'object', 'string', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()
    mejor_entidad = None
    mejor_score_separacion = 0

    for col in posibles_entidades:
        if col == col_tiempo: continue

        unicos_col = X_tr_analisis[col].nunique()
        if 1 < unicos_col < (filas_validas * 0.9): 
            duplicados_promedio = X_tr_analisis.groupby([col, col_tiempo]).size().mean()
            score_separacion = 1 / duplicados_promedio

            if score_separacion > mejor_score_separacion:
                mejor_score_separacion = score_separacion
                mejor_entidad = col

            if duplicados_promedio == 1.0:
                break 

    if mejor_entidad and mejor_score_separacion >= 0.66: 
        print(f"  ✅ [DIAGNÓSTICO] Llave de Entidad encontrada: '{mejor_entidad}'.")
        print(f"    ↳ Topología deducida: DATOS DE PANEL (Longitudinal).")
        reporte['topologia'] = 'Datos de Panel'
        reporte['columna_entidad'] = mejor_entidad
        reporte['es_serie_tiempo_estricta'] = True

        # 🚀 LA CAJA FUERTE MLOPS (Protección Absoluta + Preservación de Columna)
        indices_a_proteger = nombres_originales.copy()
        if mejor_entidad not in indices_a_proteger:
            indices_a_proteger.append(mejor_entidad)

        print(f"    🛡️ Copiando {indices_a_proteger} al Index (columna original queda intacta).")

        # 🔧 FIX MLOps: Usamos drop=False para clonar al index sin eliminar la columna
        X_tr_analisis = X_tr_analisis.set_index(indices_a_proteger, drop=False)
        if X_te_analisis is not None:
            X_te_analisis = X_te_analisis.set_index(indices_a_proteger, drop=False)

    else:
        print("  ⚠️ [DIAGNÓSTICO] No se encontró un ID claro que separe perfectamente los eventos.")
        print("    ↳ Topología deducida: TRANSVERSAL (Tratar como eventos independientes).")
        reporte['topologia'] = 'Transversal'
        reporte['es_serie_tiempo_estricta'] = False 

        if nombres_originales:
            # 🚀 FIX: Mantenemos las columnas
            X_tr_analisis = X_tr_analisis.set_index(nombres_originales, drop=False)
            if X_te_analisis is not None: X_te_analisis = X_te_analisis.set_index(nombres_originales, drop=False)

    print(f"⏱️ Diagnóstico completado en {time.time() - inicio_timer:.3f}s")
    return X_tr_analisis, X_te_analisis, reporte

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    # 🚀 FIX: Ejecutamos el diagnóstico utilizando el manager
    X_train_top, X_test_top, reporte_topologia = diagnostico_topologico_automl(
        X_train=manager.X_train, 
        X_test=manager.X_test
    )

    # Guardamos los resultados (Caja fuerte) en el Manager
    manager.X_train = X_train_top
    manager.X_test = X_test_top
    manager.rutas['reporte_topologia'] = reporte_topologia # Guardamos el reporte en rutas

    # ==========================================
    # 🔗 AUTOWIRING MLOPS: Conexión Automática a Fases Futuras
    # ==========================================
    ES_SERIE_TIEMPO_ESTRICTA = reporte_topologia['es_serie_tiempo_estricta']
    VARIABLE_TIEMPO_GLOBAL = reporte_topologia['columna_tiempo']
    VARIABLE_ENTIDAD_GLOBAL = reporte_topologia['columna_entidad']

    # Guardamos variables de enrutamiento globales directamente en el manager
    manager.rutas['es_serie_tiempo_estricta'] = ES_SERIE_TIEMPO_ESTRICTA
    manager.rutas['variable_tiempo_global'] = VARIABLE_TIEMPO_GLOBAL
    manager.rutas['variable_entidad_global'] = VARIABLE_ENTIDAD_GLOBAL

    print("\n>>> 🤖 VARIABLES DE ENRUTAMIENTO GLOBAL CONFIGURADAS EN EL MANAGER <<<")
    print(f" ⚙️ ES_SERIE_TIEMPO_ESTRICTA = {manager.rutas['es_serie_tiempo_estricta']}")
    print(f" ⚙️ VARIABLE_TIEMPO_GLOBAL = '{manager.rutas['variable_tiempo_global']}'")
    print(f" ⚙️ VARIABLE_ENTIDAD_GLOBAL = '{manager.rutas['variable_entidad_global']}'")

    # (Transición) Reflejar temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test

except Exception as e:
    print(f"🛑 Error en el Diagnóstico Topológico: {e}")


# In[80]:


X_train.head()


# In[81]:


X_train.info()


# In[82]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import copy
from typing import Tuple, Dict

def ingenieria_temporal_automl(
    X: pd.DataFrame, 
    rutas: Dict,
    fechas_ancla_aprendidas: Dict = None,
    topologia_dataset: str = 'Transversal' # 🔧 NUEVO: El Enrutador Topológico Maestro
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 3 - Paso 9.2] Motor AutoML de Ingeniería Temporal y Cíclica.
    - Sincronización MLOps: Protege el diccionario de rutas para Train y Test.
    - Transformación Circular: Usa np.sin y np.cos para codificar variables cíclicas.
    - Inteligencia Topológica (NUEVO): 
      Si es 'Transversal' -> Destruye la fecha original tras procesarla.
      Si es 'Serie de Tiempo Pura' o 'Datos de Panel' -> Preserva la fecha original para la Fase de Rezagos.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz predictora (X) está vacía.")

    print(f"=== ⏱️ FASE 9.2: Ingeniería Temporal y Trigonometría AutoML ===")
    inicio_timer = time.time()

    X_trans = X.copy()

    # Usamos deepcopy para no alterar el diccionario original inadvertidamente
    rutas_actualizadas = copy.deepcopy(rutas)

    # 🧠 INTELIGENCIA DE MEMORIA: En Train leemos las rutas, en Test leemos la memoria del Train
    if fechas_ancla_aprendidas is None:
        fechas_detectadas = rutas_actualizadas.get('date_vars', [])
    else:
        fechas_detectadas = list(fechas_ancla_aprendidas.keys())

    anclas_actuales = {} if fechas_ancla_aprendidas is None else fechas_ancla_aprendidas

    # 1. Bypass Inteligente (Escudo MLOps)
    if not fechas_detectadas:
        print("  ✅ [BYPASS] No se detectaron variables temporales para procesar.")
        print("  ⏩ La matriz permanece intacta. Avanzando al siguiente paso...")
        return X_trans, rutas_actualizadas, anclas_actuales

    # 🚀 DECISIÓN ESTRATÉGICA: ¿Destruir o Preservar?
    preservar_fecha = topologia_dataset in ['Serie de Tiempo Pura', 'Datos de Panel']

    if fechas_ancla_aprendidas is None:
        modo_str = f"{topologia_dataset.upper()} (Preservando Fecha)" if preservar_fecha else f"{topologia_dataset.upper()} (Destruyendo Fecha)"
        print(f"  🚂 [TRAIN] Procesando {len(fechas_detectadas)} variables temporales. Modo: {modo_str}")
    else:
        print(f"  🔒 [TEST] Aplicando transformaciones temporales (Sincronización exacta con Train)...")

    nuevas_numericas = []

    # 2. Motor de Extracción y Transformación
    for col in fechas_detectadas:
        if col not in X_trans.columns:
            raise KeyError(f"🛑 [Desincronización Crítica] La columna '{col}' procesada en Train no existe en Test.")

        X_trans[col] = pd.to_datetime(X_trans[col], errors='coerce')

        # A. Distancia Lineal MLOps (Antigüedad / Tendencia)
        if fechas_ancla_aprendidas is None:
            fecha_ancla = X_trans[col].max()
            anclas_actuales[col] = fecha_ancla
        else:
            fecha_ancla = fechas_ancla_aprendidas[col]

        nombre_lineal = f"{col}_antiguedad_dias"
        X_trans[nombre_lineal] = (fecha_ancla - X_trans[col]).dt.days
        nuevas_numericas.append(nombre_lineal)

        # B. Extracción de Componentes
        meses = X_trans[col].dt.month
        dias_semana = X_trans[col].dt.dayofweek

        # C. Transformación Cíclica (Seno y Coseno)
        nombre_mes_sin, nombre_mes_cos = f"{col}_mes_sin", f"{col}_mes_cos"
        X_trans[nombre_mes_sin] = np.sin(2 * np.pi * meses / 12.0)
        X_trans[nombre_mes_cos] = np.cos(2 * np.pi * meses / 12.0)

        nombre_dia_sin, nombre_dia_cos = f"{col}_dia_semana_sin", f"{col}_dia_semana_cos"
        X_trans[nombre_dia_sin] = np.sin(2 * np.pi * dias_semana / 7.0)
        X_trans[nombre_dia_cos] = np.cos(2 * np.pi * dias_semana / 7.0)

        nuevas_numericas.extend([nombre_mes_sin, nombre_mes_cos, nombre_dia_sin, nombre_dia_cos])

        # D. Inteligencia de Guillotina basada en la Topología
        if not preservar_fecha:
            X_trans.drop(columns=[col], inplace=True)
            if fechas_ancla_aprendidas is None:
                print(f"    ⚙️ '{col}' descompuesta en 5 vectores matemáticos y ELIMINADA.")
        else:
            if fechas_ancla_aprendidas is None:
                print(f"    ⚙️ '{col}' descompuesta en 5 vectores matemáticos y PRESERVADA intacta.")

    # 3. Actualización Dinámica del Enrutamiento
    if fechas_ancla_aprendidas is None:
        if not preservar_fecha:
            # Solo vaciamos la ruta de fechas si realmente la destruimos
            rutas_actualizadas['date_vars'] = [] 
        rutas_actualizadas['num_vars'].extend(nuevas_numericas)

    print("-" * 80)
    estado_col = "mantenidas vivas" if preservar_fecha else "eliminadas"
    print(f"  📊 Reporte: {len(fechas_detectadas)} columnas temporales procesadas y {estado_col}.")
    print(f"  📈 Inyectadas {len(nuevas_numericas)} nuevas variables puramente matemáticas.")
    print(f"\n⏱️ Ingeniería Temporal completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas_actualizadas, anclas_actuales


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # ---------------------------------------------------------
    # 🔗 AUTOWIRING MLOPS: Heredando Topología del Manager
    # ---------------------------------------------------------
    # Si la topología se calculó y guardó en rutas, la extraemos de ahí. Si no, asume Transversal.
    TOPOLOGIA_GLOBAL = manager.rutas.get('reporte_topologia', {}).get('topologia', 'Transversal')

    print("\n>>> 🚂 ENTRENANDO INGENIERÍA TEMPORAL EN TRAIN <<<")
    X_train_temp, rutas_actualizadas, anclas_temporales = ingenieria_temporal_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        topologia_dataset=TOPOLOGIA_GLOBAL # <- Alimentación dinámica desde el Manager
    )

    print("\n>>> 🔒 APLICANDO INGENIERÍA TEMPORAL A TEST <<<")
    X_test_temp, _, _ = ingenieria_temporal_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        fechas_ancla_aprendidas=anclas_temporales,
        topologia_dataset=TOPOLOGIA_GLOBAL # <- Alimentación dinámica desde el Manager
    )

    # Guardar en la memoria del Manager
    manager.X_train = X_train_temp
    manager.X_test = X_test_temp
    manager.rutas = rutas_actualizadas # MLOps Tip: El enrutador ya se actualizó por dentro de la función Train
    manager.guardar_artefacto('anclas_temporales', anclas_temporales)

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas
    num_vars = manager.rutas['num_vars']

except Exception as e:
    print(f"🛑 Error en la Ingeniería Temporal: {e}")


# In[83]:


X_train.info()


# In[84]:


X_test.info()


# In[85]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import re
from typing import Dict, List, Tuple

def generar_ratios_negocio_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    operaciones_ratio_manuales: List[Tuple[str, str, str]] = None,
    auto_discovery: bool = True,
    receta_aprendida: List[Tuple[str, str, str]] = None
) -> Tuple[pd.DataFrame, Dict, List[Tuple[str, str, str]]]:
    """
    [FASE 3 - Paso 9.3] Motor AutoML de Ratios Matemáticos (Universal Multi-Dominio).
    - Auto-Discovery NLP: Escanea nombres buscando magnitudes y divisores lógicos (Solo en Train).
    - Muro MLOps: Test usa estrictamente la 'receta_aprendida' de Train para garantizar alineación de columnas.
    - BLINDAJE LÉXICO: Ignora meta-variables (nulos, missing, etc.) y usa Regex para evitar falsos positivos.
    - Blindaje Anti-Infinito: Detecta divisiones por cero y neutraliza.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz predictora (X) está vacía.")

    print(f"=== ➗ FASE 9.3: Generación de Ratios Matemáticos (Auto-Discovery Universal) ===")
    inicio_timer = time.time()

    X_trans = X.copy()

    # 1. MLOps: Fit vs Transform (Alineación de Matrices)
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando receta de ratios matemáticos estricta aprendida en Train...")
        operaciones_finales = receta_aprendida
    else:
        print("  🚂 [TRAIN] Escaneando topología para Auto-Descubrimiento de Ratios...")
        operaciones_finales = operaciones_ratio_manuales or []

        if auto_discovery:
            cols_numericas = rutas.get('num_vars', X_trans.select_dtypes(include=['number']).columns.tolist())

            # ==========================================
            # 🌐 EL CEREBRO LÉXICO UNIVERSAL MULTI-DOMINIO
            # ==========================================
            def construir_patron(palabra):
                """Crea un patrón Regex para capturar palabras exactas en snake_case o camelCase"""
                return fr'(^{palabra}$|^{palabra}_|_{palabra}$|_{palabra}_|[a-z]{palabra.capitalize()})'

            # --- DICCIONARIO EXPANDIDO DE MAGNITUDES (NUMERADORES) ---
            kw_numerador = [
                # Finanzas y Ventas (Inglés/Español)
                'gain', 'ganancia', 'loss', 'perdida', 'pérdida', 'income', 'ingreso', 'ingresos', 
                'revenue', 'cost', 'costo', 'amount', 'monto', 'cantidad', 'total', 'price', 'precio', 
                'balance', 'saldo', 'sales', 'ventas', 'profit', 'beneficio', 'margin', 'margen', 
                'debt', 'deuda', 'tax', 'impuesto', 'discount', 'descuento', 'budget', 'presupuesto', 'expense', 'gasto',
                # Telemetría y Sistemas (Inglés/Español)
                'bytes', 'packets', 'paquetes', 'requests', 'peticiones', 'solicitudes', 
                'errors', 'errores', 'traffic', 'trafico', 'tráfico', 'payload', 'carga',
                # Salud y Biometría (Inglés/Español)
                'dosage', 'dosis', 'calories', 'calorias', 'calorías', 'cholesterol', 'colesterol', 
                'glucose', 'glucosa', 'heart_rate', 'frecuencia_cardiaca', 'blood_pressure', 'presion_arterial',
                # Física y Producción (Inglés/Español)
                'distance', 'distancia', 'weight', 'peso', 'production', 'produccion', 'producción', 
                'volume', 'volumen', 'length', 'longitud', 'height', 'altura', 'mass', 'masa', 
                'energy', 'energia', 'energía', 'power', 'potencia', 'yield', 'rendimiento', 'inventory', 'inventario'
            ]

            # --- DICCIONARIO EXPANDIDO DE DIVISORES (DENOMINADORES) ---
            kw_denominador = [
                # Tiempo y Duración (Inglés/Español)
                'hour', 'hora', 'day', 'dia', 'día', 'month', 'mes', 'year', 'año', 'ano', 
                'duration', 'duracion', 'duración', 'time', 'tiempo', 'seconds', 'segundos', 
                'minutes', 'minutos', 'age', 'edad', 'week', 'semana', 'quarter', 'trimestre',
                # Conteo y Capacidades (Inglés/Español)
                'qty', 'quantity', 'count', 'conteo', 'limit', 'limite', 'límite', 
                'capacity', 'capacidad', 'size', 'tamaño', 'tamano',
                # Entidades Per Cápita / Tasas (Inglés/Español)
                'users', 'usuarios', 'employees', 'empleados', 'visitors', 'visitantes', 
                'sessions', 'sesiones', 'clicks', 'clics', 'customers', 'clientes', 
                'accounts', 'cuentas', 'views', 'vistas', 'impressions', 'impresiones', 
                'transactions', 'transacciones', 'members', 'miembros', 'population', 'poblacion', 'población', 
                'area', 'área', 'capita'
            ]

            # Compilación de Regex de alto rendimiento
            patrones_numerador = [construir_patron(kw) for kw in kw_numerador]
            patron_num_regex = re.compile('|'.join(patrones_numerador), re.IGNORECASE)

            patrones_denominador = [construir_patron(kw) for kw in kw_denominador]
            patron_den_regex = re.compile('|'.join(patrones_denominador), re.IGNORECASE)

            kw_prohibidos = ['nulo', 'null', 'missing', 'tiene_']
            cols_limpias = [c for c in cols_numericas if not any(prohibido in c.lower() for prohibido in kw_prohibidos)]

            # 🚀 APLICACIÓN DEL ESCUDO LÉXICO
            nums_detectados = [c for c in cols_limpias if patron_num_regex.search(c)]
            dens_detectados = [c for c in cols_limpias if patron_den_regex.search(c)]

            for num_col in nums_detectados:
                for den_col in dens_detectados:
                    if num_col != den_col:
                        nuevo_nombre = f"{num_col}_por_{den_col}"
                        if not any(nuevo_nombre == t[0] for t in operaciones_finales):
                            operaciones_finales.append((nuevo_nombre, num_col, den_col))
                            print(f"    ✨ [Auto-Discovery] Pareja detectada: '{num_col}' / '{den_col}'")

    if not operaciones_finales:
        print("  ✅ [BYPASS] No se encontraron parejas lógicas ni ratios manuales. Avanzando...")
        return X_trans, rutas, []

    # ==========================================
    # 2. Ejecución Matemática Protegida
    # ==========================================
    nuevas_numericas = []
    ratios_creados = 0
    ratios_fallidos = 0

    print("\n  ⚙️ Ejecutando divisiones matemáticas con protección contra ceros...")

    for nuevo_nombre, col_numerador, col_denominador in operaciones_finales:
        if col_numerador not in X_trans.columns or col_denominador not in X_trans.columns:
            ratios_fallidos += 1
            continue

        numerador = X_trans[col_numerador].astype(float)
        denominador = X_trans[col_denominador].astype(float)

        # Blindaje anti-infinito: Si el denominador es 0, el resultado es 0. 
        # Si no, se divide normal. Reemplazamos 0 por nan temporalmente para evitar el warning de Pandas
        X_trans[nuevo_nombre] = np.where(
            denominador == 0, 
            0.0, 
            numerador / denominador.replace(0, np.nan)
        )

        # Limpieza extra de seguridad
        X_trans[nuevo_nombre] = X_trans[nuevo_nombre].replace([np.inf, -np.inf], 0.0)
        nuevas_numericas.append(nuevo_nombre)
        ratios_creados += 1
        print(f"    ⚖️ [Ratio Exitoso] Creada variable purgada: '{nuevo_nombre}'")

    # MLOPS TIP: Actualizamos las rutas SOLO en la ejecución de Train (la primera vez)
    if receta_aprendida is None and nuevas_numericas:
        rutas['num_vars'].extend(nuevas_numericas)

    print("-" * 80)
    print(f"  📊 Reporte: {ratios_creados} ratios inyectados | {ratios_fallidos} omitidos.")
    if ratios_creados > 0:
        print("  🛡️ ESTATUS: Gradientes protegidos. 0% de valores infinitos garantizado.")
    print(f"\n⏱️ Ingeniería de Ratios completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, operaciones_finales

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 ENTRENANDO MOTOR DE RATIOS EN TRAIN <<<")
    X_train_ratio, rutas_actualizadas, receta_ratios_train = generar_ratios_negocio_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        auto_discovery=True
    )

    print("\n>>> 🔒 APLICANDO RECETA DE RATIOS A TEST <<<")
    X_test_ratio, _, _ = generar_ratios_negocio_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        auto_discovery=False, # Bloqueamos el cerebro léxico en Test
        receta_aprendida=receta_ratios_train # Forzamos la receta de Train
    )

    # Guardamos en la memoria del Manager
    manager.X_train = X_train_ratio
    manager.X_test = X_test_ratio
    manager.rutas = rutas_actualizadas # La función Train ya inyectó las nuevas num_vars en las rutas
    manager.guardar_artefacto('receta_ratios_train', receta_ratios_train)

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas
    num_vars = manager.rutas['num_vars']

except Exception as e:
    print(f"🛑 Error en la Ingeniería de Ratios: {e}")


# In[86]:


X_train.info()


# In[87]:


X_test.info()


# In[88]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict

def mapeo_binario_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    mapeos_aprendidos: Dict = None
) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict]:
    """
    [FASE 10 - Paso 10.1] Motor AutoML de Codificación Binaria Estricta.
    - Muro MLOps: Aprende los mapeos en Train (.fit) y los aplica ciegamente en Test (.transform).
    - Escáner de Cardinalidad: Detecta automáticamente columnas con exactamente 2 valores únicos.
    - Forzado Booleano (NUEVO): Si es de tipo bool/boolean, la transforma a 0 y 1 sí o sí.
    - Escudo Missing (NUEVO): Ignora por completo las columnas generadas 'is_missing'.
    - Preservación de Nulos: Transforma a 0 y 1 dejando los NaNs intactos para la imputación posterior.
    - Downcasting Inteligente: Fuerza la conversión a int8/Int8 para optimización absoluta de RAM.
    - Integración del Target: Evalúa y codifica la variable objetivo (y) de forma estricta.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz predictora (X) está vacía.")

    print(f"=== 🎭 FASE 10.1: Mapeo Binario Estricto (Traductor AutoML) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    y_trans = y.copy() if y is not None else None
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    diccionario_mapeos = {} if mapeos_aprendidos is None else mapeos_aprendidos
    columnas_transformadas = 0

    # ==========================================
    # 1. Escáner y Traducción de Características (X)
    # ==========================================
    if mapeos_aprendidos is None:
        print("  🚂 [TRAIN] Escaneando matriz en busca de variables estrictamente binarias...")
        for col in X_trans.columns:

            # 🛡️ Escudo Missing: Ignorar columnas creadas por el imputador
            if 'is_missing' in col:
                continue

            es_booleano = pd.api.types.is_bool_dtype(X_trans[col])
            valores_unicos = X_trans[col].dropna().unique()

            # Si es numérica y ya está codificada en 0 y 1, la saltamos (a menos que sea booleano nativo)
            if not es_booleano and pd.api.types.is_numeric_dtype(X_trans[col]) and set(valores_unicos).issubset({0, 1, 0.0, 1.0}):
                continue

            # Regla del Arquitecto: Si son 2 valores exactos, o si es un booleano nativo, se transforma.
            if len(valores_unicos) == 2 or es_booleano:

                # Definición del mapa según el tipo
                if es_booleano:
                    mapa = {False: 0, True: 1}
                    mensaje_log = "False ➔ 0 | True ➔ 1"
                else:
                    valores_ordenados = sorted(list(valores_unicos))
                    mapa = {valores_ordenados[0]: 0, valores_ordenados[1]: 1}
                    mensaje_log = f"{valores_ordenados[0]} ➔ 0 | {valores_ordenados[1]} ➔ 1"

                X_trans[col] = X_trans[col].map(mapa).astype('Int8')
                diccionario_mapeos[col] = mapa
                columnas_transformadas += 1

                if col in rutas.get('cat_vars', []):
                    rutas['cat_vars'].remove(col)
                if col not in rutas.get('bool_vars', []):
                    rutas['bool_vars'].append(col)

                print(f"    🔄 [Codificado] '{col}': {mensaje_log}")
    else:
        print("  🔒 [TEST] Aplicando mapeos binarios aprendidos de Train...")
        for col, mapa in diccionario_mapeos.items():
            if not col.startswith('TARGET_') and col in X_trans.columns:
                X_trans[col] = X_trans[col].map(mapa).astype('Int8')
                columnas_transformadas += 1
                print(f"    ↳ Replicado en '{col}': {mapa}")

    # ==========================================
    # 2. Escáner y Traducción del Target (y)
    # ==========================================
    if y_trans is not None:
        if mapeos_aprendidos is None:
            valores_unicos_y = y_trans.unique()
            if len(valores_unicos_y) == 2 and not pd.api.types.is_numeric_dtype(y_trans):
                valores_ordenados_y = sorted(list(valores_unicos_y))
                mapa_y = {valores_ordenados_y[0]: 0, valores_ordenados_y[1]: 1}

                y_trans = y_trans.map(mapa_y).astype(np.int8)
                diccionario_mapeos['TARGET_' + y_trans.name] = mapa_y
                print(f"\n  🎯 [TRAIN Target] '{y_trans.name}': {valores_ordenados_y[0]} ➔ 0 | {valores_ordenados_y[1]} ➔ 1")
        else:
            clave_target = 'TARGET_' + y_trans.name
            if clave_target in diccionario_mapeos:
                mapa_y = diccionario_mapeos[clave_target]
                y_trans = y_trans.map(mapa_y).astype(np.int8)
                print(f"\n  🎯 [TEST Target] '{y_trans.name}' transformado usando: {mapa_y}")

    # ==========================================
    # 3. Reporte Ejecutivo MLOps
    # ==========================================
    print("-" * 80)
    if columnas_transformadas > 0 or (y_trans is not None and 'TARGET_' + y_trans.name in diccionario_mapeos):
        print(f"  ✅ [ESTADO SALVADO] {columnas_transformadas} características y el Target fueron binarizados.")
    else:
        print("  ⚠️ [MATRIZ LIMPIA] No se detectaron nuevas variables binarias para codificar.")

    print(f"\n⏱️ Codificación Binaria completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, y_trans, rutas, diccionario_mapeos

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None or manager.y_train is None or manager.y_test is None:
        raise ValueError("El Manager no tiene cargadas todas las matrices Train/Test completas.")

    print(">>> 🚂 ENTRENANDO MAPEO BINARIO EN TRAIN <<<")
    X_train_bin, y_train_bin, rutas_actualizadas, reglas_binarias = mapeo_binario_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas
    )

    print("\n>>> 🔒 APLICANDO MAPEO BINARIO A TEST <<<")
    X_test_bin, y_test_bin, _, _ = mapeo_binario_automl(
        X=manager.X_test, 
        y=manager.y_test, 
        rutas=manager.rutas,
        mapeos_aprendidos=reglas_binarias # El puente de memoria MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_bin
    manager.y_train = y_train_bin
    manager.X_test = X_test_bin
    manager.y_test = y_test_bin
    manager.rutas = rutas_actualizadas
    manager.guardar_artefacto('reglas_binarias', reglas_binarias)

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    y_test = manager.y_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Codificación Binaria: {e}")


# In[89]:


X_train.head()


# In[90]:


X_test.head()


# In[91]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, List

def codificador_target_multiclase(
    y: pd.Series, 
    modo: str = 'train',
    mapa_aprendido: Dict = None,
    jerarquia_ordinal: List[str] = None
) -> Tuple[pd.Series, Dict]:
    """
    [FASE 10 - Paso 10.2] Motor AutoML de Codificación de Target Multiclase y Binario.
    - Inteligencia: Procesa perfectamente tanto targets Binarios (2 clases) como Multiclase (>2).
    - Modalidad Nominal: Asigna 0, 1, 2... alfabéticamente si no hay orden.
    - Modalidad Ordinal: Respeta una lista estricta proporcionada por el Arquitecto.
    - Muro MLOps: Aprende en 'train' y aplica de forma estricta en 'test'.
    """
    if y is None or y.empty:
        raise ValueError("🛑 Error Crítico: El vector objetivo (y) está vacío.")

    print(f"=== 🎯 FASE 10.2: Codificador de Target [{modo.upper()}] ===")
    inicio_timer = time.time()

    y_trans = y.copy()

    # Bypass Inteligente: Si el target ya es numérico, no lo tocamos.
    if pd.api.types.is_numeric_dtype(y_trans):
        if modo == 'train':
            print("  ✅ [BYPASS] El Target ya es numérico. No requiere codificación.")
            return y_trans.astype(np.int8), {}
        elif modo == 'test' and not mapa_aprendido:
            print("  ✅ [BYPASS TEST] Diccionario vacío heredado. El Target se mantiene intacto.")
            return y_trans.astype(np.int8), {}

    # ==========================================
    # 1. MODO TRAIN (Aprendizaje de la Receta)
    # ==========================================
    if modo == 'train':
        valores_unicos = y_trans.dropna().unique()

        # 🚀 FIX MLOps: Manejo universal Binario/Multiclase
        if len(valores_unicos) <= 2:
            print(f"  💡 [INFO] Target BINARIO detectado ({len(valores_unicos)} clases). Codificando a 0 y 1.")
        else:
            print(f"  💡 [INFO] Target MULTICLASE detectado ({len(valores_unicos)} clases).")

        # Opción A: El Arquitecto definió un orden (Ordinal)
        if jerarquia_ordinal:
            print("  🧠 [MODO ORDINAL] Aplicando jerarquía estricta del Arquitecto...")
            # Validar que todos los valores del dataset existan en la lista del Arquitecto
            faltantes = set(valores_unicos) - set(jerarquia_ordinal)
            if faltantes:
                raise ValueError(f"🛑 Error: La jerarquía no incluye estas clases encontradas en los datos: {faltantes}")

            mapa_target = {clase: idx for idx, clase in enumerate(jerarquia_ordinal)}

        # Opción B: Automático Alfabético (Nominal)
        else:
            print("  🤖 [MODO NOMINAL] Generando mapeo alfabético automático...")
            valores_ordenados = sorted(list(valores_unicos))
            mapa_target = {clase: idx for idx, clase in enumerate(valores_ordenados)}

        # Aplicamos la transformación
        y_trans = y_trans.map(mapa_target).astype(np.int8)
        print(f"  💾 Diccionario de Mapeo Creado: {mapa_target}")
        print(f"⏱️ Target Encoding completado en {time.time() - inicio_timer:.3f}s")

        return y_trans, mapa_target

    # ==========================================
    # 2. MODO TEST (Aplicación Estricta)
    # ==========================================
    elif modo == 'test':
        if mapa_aprendido is None:
            raise ValueError("🛑 Error: En modo 'test' debes proporcionar el 'mapa_aprendido' de la fase Train.")

        if mapa_aprendido == {}:
            print("  ✅ [BYPASS TEST] Diccionario vacío heredado. El Target se mantiene intacto.")
            return y_trans, {}

        # Verificación de clases fantasma en Test (clases que no existían en Train)
        clases_test = set(y_trans.dropna().unique())
        clases_train = set(mapa_aprendido.keys())
        clases_fantasma = clases_test - clases_train

        if clases_fantasma:
            print(f"  🚨 [ALERTA MLOPS] Se detectaron clases en TEST que no existían en TRAIN: {clases_fantasma}")
            print("     ↳ Se asignará el valor especial -1 a estas clases desconocidas.")

            # Agregamos los fantasmas al mapa con valor -1 para que no se rompa el código
            for fantasma in clases_fantasma:
                mapa_aprendido[fantasma] = -1

        y_trans = y_trans.map(mapa_aprendido).astype(np.int8)
        print(f"  🔒 Replicando diccionario de Train en Test: {mapa_aprendido}")
        print(f"⏱️ Target Encoding completado en {time.time() - inicio_timer:.3f}s")

        return y_trans, mapa_aprendido

    else:
        raise ValueError("🛑 El modo debe ser 'train' o 'test'.")

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.y_train is None or manager.y_test is None:
        raise ValueError("El Manager no tiene cargados los vectores 'y_train' o 'y_test'.")

    # 🛑 SWITCH DEL ARQUITECTO
    # Si tu target tiene un orden lógico (ej. 'Bajo', 'Medio', 'Alto'), escríbelo aquí en orden.
    # Si no tiene orden (ej. 'Perro', 'Gato', 'Pájaro'), déjalo como None.
    MI_JERARQUIA_TARGET = None 
    # Ejemplo de uso: MI_JERARQUIA_TARGET = ['Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto']

    print(">>> 🚂 ENTRENANDO TARGET MULTICLASE <<<")
    y_train_mc, diccionario_target_maestro = codificador_target_multiclase(
        y=manager.y_train, 
        modo='train',
        jerarquia_ordinal=MI_JERARQUIA_TARGET
    )

    print("\n>>> 🔒 APLICANDO A TEST <<<")
    y_test_mc, _ = codificador_target_multiclase(
        y=manager.y_test, 
        modo='test',
        mapa_aprendido=diccionario_target_maestro
    )

    # Guardamos los resultados en el Manager
    manager.y_train = y_train_mc
    manager.y_test = y_test_mc
    manager.guardar_artefacto('diccionario_target_maestro', diccionario_target_maestro)

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    y_train = manager.y_train
    y_test = manager.y_test

except Exception as e:
    print(f"🛑 Error en el codificador de Target: {e}")


# In[92]:


y_train.head()


# In[93]:


y_test.head()


# In[94]:


X_train.head()


# In[95]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
from typing import Tuple, Dict

def codificacion_ordinal_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    diccionarios_manuales: Dict[str, Dict[str, int]] = None,
    receta_aprendida: Dict[str, Dict[str, int]] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 10 - Paso 10.3] Motor AutoML de Codificación Ordinal.
    - Híbrido MLOps: En Train combina manual + auto-discovery. En Test solo aplica receta.
    - Preservación de Nulos: Usa .map() puro, garantizando que los NaNs sigan siendo NaNs.
    - MLOps State: Retorna el artefacto de traducción para el despliegue en Producción.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz predictora (X) está vacía.")

    print(f"=== 📶 FASE 10.3: Codificación Ordinal Jerárquica (Motor Híbrido) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': []}
    columnas_transformadas = 0

    # ==========================================
    # 1. El Cerebro NLP y Mapeo MLOps (Fit vs Transform)
    # ==========================================
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando jerarquías estrictas aprendidas en Train...")
        diccionarios_finales = receta_aprendida
    else:
        print("  🚂 [TRAIN] Buscando jerarquías de negocio y fusionando manuales...")
        diccionarios_finales = diccionarios_manuales or {}

        columnas_texto = [col for col in X_trans.columns if col in rutas.get('cat_vars', X_trans.select_dtypes(include=['object', 'category']).columns)]

        jerarquias_universales = {
            'niveles_basicos': {'low': 1, 'medium': 2, 'high': 3},
            'tallas_ropa': {'s': 1, 'm': 2, 'l': 3, 'xl': 4, 'xxl': 5},
            'calidad': {'bad': 1, 'poor': 2, 'fair': 3, 'good': 4, 'excellent': 5}
        }

        for col in columnas_texto:
            if col in diccionarios_finales:
                continue

            valores_unicos = set(X_trans[col].dropna().astype(str).str.lower())

            for nombre_jerarquia, diccionario_nlp in jerarquias_universales.items():
                claves_nlp = set(diccionario_nlp.keys())
                interseccion = valores_unicos.intersection(claves_nlp)

                if len(valores_unicos) > 0 and len(interseccion) / len(valores_unicos) >= 0.8:
                    mapa_auto = {}
                    for val_real in X_trans[col].dropna().unique():
                        val_lower = str(val_real).lower()
                        if val_lower in diccionario_nlp:
                            mapa_auto[val_real] = diccionario_nlp[val_lower]

                    diccionarios_finales[col] = mapa_auto
                    print(f"    ✨ [Auto-Discovery] Detectada jerarquía '{nombre_jerarquia}' en '{col}'.")
                    break

    # ==========================================
    # 2. Motor de Traducción Blindada
    # ==========================================
    if not diccionarios_finales:
        print("  ✅ [BYPASS] No se definieron jerarquías manuales ni se auto-detectaron patrones universales.")
        return X_trans, rutas, {}

    print("\n  ⚙️ Aplicando mapeo explícito de enteros...")

    for col, mapa in diccionarios_finales.items():
        if col in X_trans.columns:
            # .map() reemplaza por NaN todo lo que no esté en el diccionario
            X_trans[col] = X_trans[col].map(mapa)

            # Casteamos a float temporalmente para soportar los NaNs en Pandas
            X_trans[col] = X_trans[col].astype(float)

            columnas_transformadas += 1
            print(f"    🔄 [Codificado] '{col}' convertida a enteros secuenciales.")

            # Actualizamos rutas SOLO la primera vez (en Train)
            if receta_aprendida is None:
                if col in rutas.get('cat_vars', []):
                    rutas['cat_vars'].remove(col)
                if col not in rutas.get('num_vars', []):
                    rutas['num_vars'].append(col)

    # ==========================================
    # 3. Reporte Ejecutivo MLOps
    # ==========================================
    print("-" * 80)
    print(f"  📊 Reporte: {columnas_transformadas} características transformadas exitosamente.")
    print(f"\n⏱️ Codificación Ordinal completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, diccionarios_finales

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # 💡 Lógica de Negocio (Domain Knowledge) inyectada por el Arquitecto
    mis_jerarquias = {
        'recsupervisionleveltext': {
            'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4
        }
    }

    print(">>> 🚂 ENTRENANDO MOTOR ORDINAL EN TRAIN <<<")
    X_train_ord, rutas_actualizadas, receta_ordinal = codificacion_ordinal_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        diccionarios_manuales=mis_jerarquias
    )

    print("\n>>> 🔒 APLICANDO RECETA ORDINAL A TEST <<<")
    X_test_ord, _, _ = codificacion_ordinal_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        receta_aprendida=receta_ordinal # El puente MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_ord
    manager.X_test = X_test_ord
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Usamos getattr en lugar de chequeo directo para asegurar la inicialización si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['receta_ordinal'] = receta_ordinal

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Codificación Ordinal: {e}")


# In[96]:


X_train.head()


# In[97]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict
from sklearn.model_selection import KFold

# ==========================================
# MOTOR 1: TARGET ENCODING
# ==========================================
def target_encoding_oof_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None, 
    m_suavizado: float = 10.0,
    n_splits: int = 5,
    receta_aprendida: Dict = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 10 - Paso 10.4] Motor AutoML de Target Encoding (Universal Multi-Clase + OOF).
    - Auto-Detección: Soporta Target Binario (1 prob/col) o Multiclase (N probs/col).
    - Muro MLOps: En Train calcula medias y guarda receta. En Test solo aplica receta.
    - FIX MLOps: Ignora el Index temporalmente para evadir el error de "duplicate labels".
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🎯 FASE 10.4: Target Encoding Universal (OOF + Bayesiano) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # 🚀 FIX PANDAS: Guardamos el Index original y lo ignoramos (reset_index) para la matemática
    original_index = X_trans.index
    X_trans = X_trans.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)

    # ==========================================
    # 1. Modo TEST (.transform)
    # ==========================================
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando probabilidades Bayesianas aprendidas de Train...")

        for col_original, config_encoding in receta_aprendida.items():
            if col_original not in X_trans.columns: continue

            for nombre_clase, diccionario_mapeo in config_encoding.items():
                media_global_train = diccionario_mapeo.pop('__GLOBAL_MEAN__', 0)

                nueva_col_nombre = col_original if len(config_encoding) == 1 else f"{col_original}_prob_{nombre_clase}"

                mask_nan = X_trans[col_original].isna()
                X_trans[nueva_col_nombre] = X_trans[col_original].astype(object).map(diccionario_mapeo).fillna(media_global_train)
                X_trans.loc[mask_nan, nueva_col_nombre] = np.nan

            if len(config_encoding) > 1:
                X_trans.drop(columns=[col_original], inplace=True)

        print(f"\n⏱️ Target Encoding completado en {time.time() - inicio_timer:.3f}s")

        # 🚀 RESTAURAR INDEX
        X_trans.index = original_index
        return X_trans, rutas, receta_aprendida

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if y is None: raise ValueError("🛑 Error: Train requiere la variable objetivo 'y'.")

    # --- AUTO-DETECCIÓN DEL TIPO DE TARGET ---
    valores_target = y.dropna().unique()
    es_multiclase = len(valores_target) > 2 or not pd.api.types.is_numeric_dtype(y)

    targets_a_procesar = {}
    if es_multiclase:
        print(f"  🧠 [AUTO-DETECCIÓN] Target Multiclase detectado ({len(valores_target)} categorías).")
        for clase in valores_target:
            targets_a_procesar[clase] = (y == clase).astype(float)
    else:
        print(f"  🧠 [AUTO-DETECCIÓN] Target Binario Numérico detectado.")
        targets_a_procesar['Target_Directo'] = y.copy().astype(float)

    diccionario_produccion = {}

    cols_a_codificar = []

    # --- FILTRO SILENCIOSO DE VARIABLES ---
    for c in X_trans.columns:
        if c.startswith('TARGET_'): continue

        n_unicos = X_trans[c].dropna().nunique()
        es_numerica = pd.api.types.is_numeric_dtype(X_trans[c])
        es_categoria_pura = c in rutas.get('cat_vars', []) or pd.api.types.is_object_dtype(X_trans[c]) or pd.api.types.is_categorical_dtype(X_trans[c])

        # Solo pasa el filtro si es categórica pura, no es numérica y tiene más de 2 categorías
        if es_categoria_pura and n_unicos > 2 and not es_numerica:
            cols_a_codificar.append(c)

    if not cols_a_codificar:
        print("\n  ✅ [BYPASS] No hay variables categóricas de alta cardinalidad para Target Encoding.")
        X_trans.index = original_index
        return X_trans, rutas, {}

    print(f"\n  🚂 [TRAIN] Procesando {len(cols_a_codificar)} variables con alta cardinalidad: {cols_a_codificar}...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cols_a_codificar:
        diccionario_produccion[col] = {}
        X_trans[col] = X_trans[col].astype(object)

        for nombre_clase, y_clase in targets_a_procesar.items():
            media_global = y_clase.mean()
            nueva_col = np.full(len(X_trans), np.nan)

            stats_globales = pd.DataFrame({'Target': y_clase, 'Categoria': X_trans[col]}).groupby('Categoria')['Target'].agg(['count', 'mean'])
            n_global = stats_globales['count']
            suavizado_global = (n_global * stats_globales['mean'] + m_suavizado * media_global) / (n_global + m_suavizado)

            diccionario_produccion[col][nombre_clase] = suavizado_global.to_dict()
            diccionario_produccion[col][nombre_clase]['__GLOBAL_MEAN__'] = media_global 

            for train_idx, val_idx in kf.split(X_trans):
                X_tr_fold, X_val_fold = X_trans.iloc[train_idx], X_trans.iloc[val_idx]
                y_tr_fold = y_clase.iloc[train_idx]

                stats_fold = pd.DataFrame({'Target': y_tr_fold, 'Categoria': X_tr_fold[col]}).groupby('Categoria')['Target'].agg(['count', 'mean'])
                n = stats_fold['count']
                suavizado_fold = (n * stats_fold['mean'] + m_suavizado * media_global) / (n + m_suavizado)

                nueva_col[val_idx] = X_val_fold[col].map(suavizado_fold).astype(float).fillna(media_global)

            mask_nan = X_trans[col].isna()
            nueva_col_nombre = col if not es_multiclase else f"{col}_prob_{nombre_clase}"
            X_trans[nueva_col_nombre] = nueva_col
            X_trans.loc[mask_nan, nueva_col_nombre] = np.nan

            print(f"    🔄 [Encoded] '{nueva_col_nombre}' (Media global: {media_global:.4f})")

            if nueva_col_nombre not in rutas['num_vars']:
                rutas['num_vars'].append(nueva_col_nombre)

        if es_multiclase:
            X_trans.drop(columns=[col], inplace=True)
            if col in rutas['cat_vars']: rutas['cat_vars'].remove(col)
        elif col in rutas['cat_vars']:
             rutas['cat_vars'].remove(col) 

    print("-" * 80)
    print("  🛡️ ESTATUS: Prevención de Fuga de Datos (OOF) aplicada exitosamente.")
    print(f"\n⏱️ Target Encoding completado en {time.time() - inicio_timer:.3f}s")

    # 🚀 RESTAURAR INDEX
    X_trans.index = original_index
    return X_trans, rutas, diccionario_produccion


# ==========================================
# MOTOR 2: WEIGHT OF EVIDENCE (WoE)
# ==========================================
def woe_encoding_oof_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None, 
    n_splits: int = 5,
    epsilon: float = 0.001,
    receta_aprendida: Dict = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 10 - Paso 10.4] Motor AutoML de Weight of Evidence (WoE + OOF).
    - Muro MLOps: En Train (.fit) calcula WoE OOF y guarda la receta. En Test (.transform) solo aplica la receta.
    - Matemática Segura (Epsilon): Evita divisiones por cero y logaritmos infinitos.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 10.4: Weight of Evidence - WoE (OOF + Escudo Epsilon) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # 🚀 FIX PANDAS: Guardamos el Index original y lo ignoramos (reset_index) para la matemática
    original_index = X_trans.index
    X_trans = X_trans.reset_index(drop=True)
    if y is not None:
        y = y.reset_index(drop=True)

    # ==========================================
    # 1. Modo TEST (.transform)
    # ==========================================
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando logaritmos WoE fijos aprendidos de Train...")
        columnas_a_transformar = list(receta_aprendida.keys())

        for col in columnas_a_transformar:
            if col in X_trans.columns:
                diccionario_columna = receta_aprendida[col]
                valor_neutral_train = diccionario_columna.pop('__GLOBAL_NEUTRAL__', 0.0)

                X_trans[col] = X_trans[col].astype(object)

                mask_nan = X_trans[col].isna()
                X_trans[col] = X_trans[col].map(diccionario_columna).fillna(valor_neutral_train)
                X_trans.loc[mask_nan, col] = np.nan

                print(f"    ↳ Replicado en '{col}' (Categorías nuevas llenadas con WoE Neutral: 0.0)")

        print(f"\n⏱️ WoE completado en {time.time() - inicio_timer:.3f}s")

        # 🚀 RESTAURAR INDEX
        X_trans.index = original_index
        return X_trans, rutas, receta_aprendida

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if y is None:
        raise ValueError("🛑 Error: En modo Train (sin receta aprendida) debes proporcionar 'y'.")

    y_trans = y.copy().astype(float) 
    diccionario_produccion = {}

    cols_a_codificar = []

    # --- FILTRO SILENCIOSO DE VARIABLES ---
    for c in X_trans.columns:
        if c.startswith('TARGET_'): continue

        n_unicos = X_trans[c].dropna().nunique()
        es_numerica = pd.api.types.is_numeric_dtype(X_trans[c])
        es_categoria_pura = c in rutas.get('cat_vars', []) or pd.api.types.is_object_dtype(X_trans[c]) or pd.api.types.is_categorical_dtype(X_trans[c])

        if es_categoria_pura and n_unicos > 2 and not es_numerica:
            cols_a_codificar.append(c)

    if not cols_a_codificar:
        print("\n  ✅ [BYPASS] No hay variables categóricas candidatas restantes para WoE.")
        X_trans.index = original_index
        return X_trans, rutas, {}

    print(f"\n  🚂 [TRAIN] Procesando {len(cols_a_codificar)} variables con alta cardinalidad: {cols_a_codificar}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    global_pos = y_trans.sum()
    global_neg = len(y_trans) - global_pos

    for col in cols_a_codificar:
        X_trans[col] = X_trans[col].astype(object)
        nueva_col = np.zeros(len(X_trans))
        nueva_col[:] = np.nan

        stats_globales = pd.DataFrame({'Target': y_trans, 'Categoria': X_trans[col]}).groupby('Categoria')['Target'].agg(['sum', 'count'])
        cat_pos = stats_globales['sum']
        cat_neg = stats_globales['count'] - cat_pos

        prop_pos_global = (cat_pos + epsilon) / (global_pos + epsilon * 2)
        prop_neg_global = (cat_neg + epsilon) / (global_neg + epsilon * 2)

        woe_global = np.log(prop_pos_global / prop_neg_global)

        diccionario_produccion[col] = woe_global.to_dict()
        diccionario_produccion[col]['__GLOBAL_NEUTRAL__'] = 0.0 

        for train_idx, val_idx in kf.split(X_trans):
            X_tr_fold, X_val_fold = X_trans.iloc[train_idx], X_trans.iloc[val_idx]
            y_tr_fold = y_trans.iloc[train_idx]

            fold_pos = y_tr_fold.sum()
            fold_neg = len(y_tr_fold) - fold_pos

            stats_fold = pd.DataFrame({'Target': y_tr_fold, 'Categoria': X_tr_fold[col]}).groupby('Categoria')['Target'].agg(['sum', 'count'])
            f_cat_pos = stats_fold['sum']
            f_cat_neg = stats_fold['count'] - f_cat_pos

            f_prop_pos = (f_cat_pos + epsilon) / (fold_pos + epsilon * 2)
            f_prop_neg = (f_cat_neg + epsilon) / (fold_neg + epsilon * 2)

            woe_fold = np.log(f_prop_pos / f_prop_neg)

            mapeo_val = X_val_fold[col].map(woe_fold).astype(float)
            mapeo_val = mapeo_val.fillna(0.0) 

            nueva_col[val_idx] = mapeo_val

        mask_nan = X_trans[col].isna()
        X_trans[col] = nueva_col 
        X_trans.loc[mask_nan, col] = np.nan 

        print(f"    🔄 [WoE Encoded] '{col}' transformada a Weight of Evidence (OOF).")

        if col in rutas['cat_vars']:
            rutas['cat_vars'].remove(col)
        if col not in rutas['num_vars']:
            rutas['num_vars'].append(col)

    print("-" * 80)
    print("  🛡️ ESTATUS: WoE calculado con blindaje OOF y Epsilon Anti-Infinitos.")
    print("  💾 Diccionario de mapeo guardado para la API de Producción.")
    print(f"\n⏱️ WoE completado en {time.time() - inicio_timer:.3f}s")

    # 🚀 RESTAURAR INDEX
    X_trans.index = original_index
    return X_trans, rutas, diccionario_produccion


# ==========================================
# Celda de Ejecución Unificada en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # 🛑 SWITCH MAESTRO DE CODIFICACIÓN (El Enrutador del Arquitecto)
    # True = Usa Target Encoding (Para Regresión, Árboles o Casos Generales)
    # False = Usa Weight of Evidence (Para Riesgo Crediticio / Regresión Logística Binaria)
    USAR_TARGET_ENCODING = True  

    if USAR_TARGET_ENCODING:
        print(">>> 🎯 MODO SELECCIONADO: TARGET ENCODING <<<")
        print(">>> 🚂 ENTRENANDO TARGET ENCODER EN TRAIN <<<")
        X_train_enc, rutas_actualizadas, receta_codificacion = target_encoding_oof_automl(
            X=manager.X_train, 
            y=manager.y_train, 
            rutas=manager.rutas,
            m_suavizado=10.0,
            n_splits=5
        )

        print("\n>>> 🔒 APLICANDO TARGET ENCODER A TEST <<<")
        X_test_enc, _, _ = target_encoding_oof_automl(
            X=manager.X_test, 
            y=None, 
            rutas=manager.rutas,
            receta_aprendida=receta_codificacion
        )
    else:
        print(">>> ⚖️ MODO SELECCIONADO: WEIGHT OF EVIDENCE (WoE) <<<")
        print(">>> 🚂 ENTRENANDO WoE EN TRAIN <<<")
        X_train_enc, rutas_actualizadas, receta_codificacion = woe_encoding_oof_automl(
            X=manager.X_train, 
            y=manager.y_train, 
            rutas=manager.rutas,
            n_splits=5
        )

        print("\n>>> 🔒 APLICANDO WoE A TEST <<<")
        X_test_enc, _, _ = woe_encoding_oof_automl(
            X=manager.X_test, 
            y=None, 
            rutas=manager.rutas,
            receta_aprendida=receta_codificacion
        )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_enc
    manager.X_test = X_test_enc
    manager.rutas = rutas_actualizadas

    nombre_artefacto = 'receta_target_encoding' if USAR_TARGET_ENCODING else 'receta_woe_encoding'
    manager.modelos_preprocesamiento[nombre_artefacto] = receta_codificacion

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Codificación Supervisada: {e}")


# In[98]:


X_train.head()


# In[99]:


X_test.head()


# In[100]:


y_train.head()


# In[101]:


y_test.head()


# In[102]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict

def codificacion_nativa_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    receta_aprendida: Dict = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 10 - Paso 10.5] Motor AutoML de Categorías Nativas.
    - Especial para CatBoost / LightGBM.
    - Muro MLOps: Aprende el universo en Train y lo guarda en la receta. Test solo obedece la receta.
    - Blindaje Test/Producción: Si llega una categoría nueva, la neutraliza a NaN sin caerse.
    - Reducción de Memoria: El tipo 'category' usa punteros enteros (hiper-ligero).
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🏷️ FASE 10.5: Native Categoricals (CatBoost/LightGBM Ready) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}
    columnas_transformadas = 0

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando moldes categóricos estrictos aprendidos de Train...")

        for col, categorias_conocidas in receta_aprendida.items():
            if col in X_trans.columns:
                molde_categorico = pd.CategoricalDtype(categories=categorias_conocidas, ordered=False)

                # Protegemos NaNs reales, convertimos a string y aplicamos el molde de Train
                X_trans[col] = X_trans[col].astype(str).replace('nan', np.nan)
                X_trans[col] = X_trans[col].astype(molde_categorico)

                # MAGIA MLOPS: Si X_test tenía una ciudad "Quito" que no estaba en Train, 
                # Pandas automáticamente la convierte en NaN sin lanzar error.

                columnas_transformadas += 1
                print(f"    ↳ Replicado en '{col}' (Categorías desconocidas neutralizadas a NaN).")

        print(f"\n⏱️ Tipado Nativo completado en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, receta_aprendida

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    diccionario_produccion = {}

    # Filtro Inteligente: Solo tocamos lo que "sobrevivió" a las codificaciones anteriores
    cols_a_codificar = rutas.get('cat_vars', [])

    if not cols_a_codificar:
        print("  ✅ [BYPASS] No quedan variables de texto libres. Todas fueron numéricamente codificadas.")
        return X_trans, rutas, {}

    print(f"  🚂 [TRAIN] Detectadas {len(cols_a_codificar)} variables residuales para tipado nativo: {cols_a_codificar}")

    # Motor de Tipado MLOps
    for col in cols_a_codificar:
        if col not in X_trans.columns:
            continue

        # A. Extracción del Universo Conocido
        categorias_conocidas = X_trans[col].dropna().astype(str).unique()

        # B. Creación del "Molde Estricto"
        molde_categorico = pd.CategoricalDtype(categories=categorias_conocidas, ordered=False)
        diccionario_produccion[col] = list(categorias_conocidas)

        # C. Aplicación a Train
        X_trans[col] = X_trans[col].astype(str).replace('nan', np.nan) 
        X_trans[col] = X_trans[col].astype(molde_categorico)

        columnas_transformadas += 1
        print(f"    🔄 [Tipado Nativo] '{col}' convertida a 'category' (Memoria Optimizada).")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: {columnas_transformadas} características blindadas con molde categórico estricto.")
    print("  💾 Diccionario de universos permitidos guardado para la API de Producción.")
    print(f"\n⏱️ Tipado Nativo completado en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, diccionario_produccion

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 ENTRENANDO TIPADO NATIVO EN TRAIN <<<")
    X_train_nat, rutas_actualizadas, receta_categorias = codificacion_nativa_automl(
        X=manager.X_train, 
        rutas=manager.rutas
    )

    print("\n>>> 🔒 APLICANDO TIPADO NATIVO A TEST <<<")
    X_test_nat, _, _ = codificacion_nativa_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        receta_aprendida=receta_categorias # Puente de Producción
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_nat
    manager.X_test = X_test_nat
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Asegurar inicialización si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['receta_categorias_nativas'] = receta_categorias

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en el Tipado Nativo: {e}")


# # FASE 4: Imputación, Outliers y Escalamiento
# Ahora que todo es numérico, reparamos la topología del espacio vectorial.

# In[103]:


X_train.head()


# In[104]:


X_test[10:20]


# In[105]:


X_train.info()


# In[106]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, Optional
from sklearn.impute import KNNImputer

def imputacion_knn_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    n_vecinos: int = 5,
    imputador_entrenado: Optional[KNNImputer] = None
) -> Tuple[pd.DataFrame, Dict, Optional[KNNImputer]]:
    """
    [FASE 4 - Paso 11.1] Motor AutoML de Restauración Espacial (KNN Imputer Escalable).
    - Escudo Autónomo Temporal (Clean Code): Protege fechas nativas y las de `rutas['date_vars']` sin parámetros manuales.
    - Cazador de Anomalías: Solo busca y destruye NaTs infiltrados en variables NO temporales.
    - Arquitectura Big Data: Donor Pool (max 15k) y Chunking (10k) para proteger la RAM.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🩹 FASE 11.1: Restauración Espacial (KNN Imputer Escalable - {n_vecinos} Vecinos) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    idx = X_trans.index
    total_filas = len(X_trans)

    # 🚀 INTELIGENCIA AUTO-ML: Recuperar fechas protegidas desde la memoria global
    columnas_fecha_intactas = rutas.get('date_vars', [])
    fechas_protegidas_encontradas = []

    # ==========================================
    # 🚀 PRE-PROCESO: Traducción NaT -> NaN (Exclusivo para No-Fechas)
    # ==========================================
    for col in X_trans.columns:
        # 1. 🛡️ ESCUDO AUTOMÁTICO: Si es fecha (por tipo o por ruta), la ignoramos por completo
        if pd.api.types.is_datetime64_any_dtype(X_trans[col]) or col in columnas_fecha_intactas:
            fechas_protegidas_encontradas.append(col)
            continue

        # 2. 🧠 BÚSQUEDA INTELIGENTE: Solo revisamos variables de texto/objeto que tengan nulos
        if X_trans[col].hasnans and X_trans[col].dtype == 'object':
            # Evaluamos silenciosamente si hay NaTs infiltrados (anomalía de Pandas)
            mascara_nat = X_trans[col].apply(lambda x: x is pd.NaT)
            hallazgos_nat = mascara_nat.sum()

            if hallazgos_nat > 0:
                print(f"  ⚙️ [PURIFICACIÓN] Aniquilando {hallazgos_nat} NaTs infiltrados en la variable no-temporal '{col}' -> NaN.")
                X_trans.loc[mascara_nat, col] = np.nan

    # 📊 Telemetría del Escudo Temporal
    if fechas_protegidas_encontradas:
        print(f"  🛡️ [ESCUDO ACTIVO] Se protegieron {len(fechas_protegidas_encontradas)} columnas de tipo fecha: {fechas_protegidas_encontradas}")

    # ==========================================
    # 🛡️ AISLAMIENTO QUIRÚRGICO: Solo pasamos números al KNN
    # (Las fechas protegidas quedan fuera automáticamente)
    # ==========================================
    cols_numericas = X_trans.select_dtypes(include=[np.number]).columns.tolist()

    if not cols_numericas:
        print("  ✅ [BYPASS] No se detectaron variables numéricas. Imputación omitida.")
        return X_trans, rutas, imputador_entrenado or KNNImputer()

    nulos_numericos = X_trans[cols_numericas].isna().sum().sum()

    # ==========================================
    # ⚙️ Parámetros de Escalabilidad (Big Data)
    # ==========================================
    MAX_FIT_SAMPLES = 15000  
    CHUNK_SIZE = 10000       

    def transformar_por_lotes(imputador, df_a_imputar):
        matrices_limpias = []
        for i in range(0, len(df_a_imputar), CHUNK_SIZE):
            chunk = df_a_imputar.iloc[i:i+CHUNK_SIZE].astype(float)
            chunk_imputado = imputador.transform(chunk)
            matrices_limpias.append(chunk_imputado)
        return np.vstack(matrices_limpias)

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if imputador_entrenado is not None:
        if nulos_numericos == 0:
            print("  ✅ [BYPASS] Test no tiene valores NaN en numéricas. Matriz intacta.")
        else:
            print(f"  🔒 [TEST] Rellenando {nulos_numericos:,} huecos (Chunking)...")
            matriz_imputada = transformar_por_lotes(imputador_entrenado, X_trans[cols_numericas])
            df_imputado = pd.DataFrame(matriz_imputada, columns=cols_numericas, index=idx)
            X_trans[cols_numericas] = df_imputado
            print(f"  📊 Huecos numéricos restantes en Test: {X_trans[cols_numericas].isna().sum().sum()}")

        print(f"\n⏱️ Imputación KNN completada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, imputador_entrenado

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if nulos_numericos == 0:
        print("  ✅ [BYPASS] Train no tiene valores NaN. Entrenando KNN pasivo.")
        X_fit = X_trans[cols_numericas].head(MAX_FIT_SAMPLES)
        imputador = KNNImputer(n_neighbors=n_vecinos, weights='distance')
        imputador.fit(X_fit.astype(float))
    else:
        print(f"  🚂 [TRAIN] Detectados {nulos_numericos:,} huecos numéricos. Protegiendo RAM...")

        X_num_trans = X_trans[cols_numericas]
        if total_filas > MAX_FIT_SAMPLES:
            print(f"    ↳ Dataset masivo detectado. Creando 'Donor Pool' aleatorio de {MAX_FIT_SAMPLES:,} filas...")
            X_fit = X_num_trans.sample(n=MAX_FIT_SAMPLES, random_state=42)
        else:
            X_fit = X_num_trans

        imputador = KNNImputer(n_neighbors=n_vecinos, weights='distance')
        print(f"    ↳ Construyendo topología matemática (fit)...")
        imputador.fit(X_fit.astype(float))

        print(f"    ↳ Rellenando huecos en matriz completa por lotes de {CHUNK_SIZE:,} filas (transform)...")
        matriz_imputada = transformar_por_lotes(imputador, X_num_trans)
        df_imputado = pd.DataFrame(matriz_imputada, columns=cols_numericas, index=idx)

        X_trans[cols_numericas] = df_imputado

    print("-" * 80)
    print("  🛡️ ESTATUS: Cirugía completada con Arquitectura Big Data.")
    print(f"  📊 Huecos numéricos restantes en Train: {X_trans[cols_numericas].isna().sum().sum()}")
    print("  💾 Modelo Imputador guardado para la API de Producción.")
    print(f"\n⏱️ Imputación KNN completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, imputador

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # 🚀 Clean Code Absoluto: Función 100% Autónoma, lee la memoria sola.
    print(">>> 🚂 ENTRENANDO IMPUTADOR KNN EN TRAIN <<<")
    X_train_knn, rutas_actualizadas, modelo_knn = imputacion_knn_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        n_vecinos=5
    )

    print("\n>>> 🔒 APLICANDO IMPUTADOR KNN A TEST <<<")
    X_test_knn, _, _ = imputacion_knn_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        n_vecinos=5,
        imputador_entrenado=modelo_knn 
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_knn
    manager.X_test = X_test_knn
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Asegurar inicialización de la caja fuerte de modelos si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['imputador_knn'] = modelo_knn

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Imputación KNN: {e}")


# In[107]:


X_train.head()


# In[108]:


X_test[10:20]


# In[109]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, Optional
from sklearn.ensemble import IsolationForest

def deteccion_outliers_aislamiento(
    X_train: pd.DataFrame, 
    rutas: Dict, 
    y_train: Optional[pd.Series] = None,
    X_test: Optional[pd.DataFrame] = None,
    contamination: float = 'auto',
    eliminar_en_train: bool = False,
    modelo_entrenado: Optional[IsolationForest] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series], Dict, IsolationForest]:
    """
    [FASE 4 - Paso 12.1] Motor AutoML de Detección Multivariada (Isolation Forest).
    - Tratamiento Asimétrico MLOps: Test NUNCA elimina filas, solo hereda la bandera de anomalía.
    - Filtro de Tipos: Solo escanea variables numéricas/booleanas para evitar colisiones con tipado nativo.
    - Feature Engineering: Crea la bandera 'is_anomaly_isoforest' (1 = Outlier, 0 = Inlier).
    - Purga Opcional: Si eliminar_en_train=True, destruye los outliers de X_train y y_train.
    """
    if X_train is None or X_train.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X_train) está vacía.")

    print(f"=== 🛸 FASE 12.1: Detección Multivariada Asimétrica (Isolation Forest) ===")
    inicio_timer = time.time()

    X_tr_trans = X_train.copy()
    X_te_trans = X_test.copy() if X_test is not None else None
    y_tr_trans = y_train.copy() if y_train is not None else None

    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}
    nombre_bandera = 'is_anomaly_isoforest'

    # 1. Escudo de Tipos (Solo usamos las rutas numéricas para la matemática del bosque)
    cols_matematicas = rutas.get('num_vars', []) + rutas.get('bool_vars', [])

    # 🛑 FIX QUIRÚRGICO: Evitamos que busque la propia bandera como si fuera variable predictora
    cols_validas = [c for c in cols_matematicas if c in X_tr_trans.columns and c != nombre_bandera]

    if not cols_validas:
        print("  ✅ [BYPASS] No se encontraron columnas numéricas válidas para Isolation Forest.")
        return X_tr_trans, X_te_trans, y_tr_trans, rutas, modelo_entrenado

    # ==========================================
    # 2. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if modelo_entrenado is not None:
        if X_te_trans is None:
            return X_tr_trans, X_te_trans, y_tr_trans, rutas, modelo_entrenado

        print("  🔒 [TEST] Escaneando Producción en busca de anomalías usando el Bosque de Train...")
        # Isolation Forest devuelve -1 para outliers y 1 para inliers. Lo mapeamos a 1 y 0 (int8).
        preds_test = modelo_entrenado.predict(X_te_trans[cols_validas].fillna(0)) # IF no soporta NaNs, si quedara alguno, fallback a 0
        X_te_trans[nombre_bandera] = np.where(preds_test == -1, 1, 0).astype(np.int8)

        outliers_test = X_te_trans[nombre_bandera].sum()
        print(f"    ↳ Detectados {outliers_test} extraterrestres en Test (Marcados, NUNCA eliminados).")
        print(f"\n⏱️ Escáner completado en {time.time() - inicio_timer:.3f}s")
        return X_tr_trans, X_te_trans, y_tr_trans, rutas, modelo_entrenado

    # ==========================================
    # 3. Modo TRAIN (.fit)
    # ==========================================
    print(f"  🚂 [TRAIN] Entrenando Bosque de Aislamiento sobre {len(cols_validas)} dimensiones...")

    # n_jobs=-1 usa todos los núcleos del procesador para velocidad extrema
    bosque = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

    # Entrenamos y predecimos sobre Train
    preds_train = bosque.fit_predict(X_tr_trans[cols_validas].fillna(0))
    X_tr_trans[nombre_bandera] = np.where(preds_train == -1, 1, 0).astype(np.int8)

    outliers_train = X_tr_trans[nombre_bandera].sum()
    porcentaje = (outliers_train / len(X_tr_trans)) * 100

    print(f"    ↳ Detectadas {outliers_train} anomalías multivariadas ({porcentaje:.2f}% de la matriz).")

    # Actualización de Rutas
    if nombre_bandera not in rutas.get('bool_vars', []):
        rutas['bool_vars'].append(nombre_bandera)

    # 4. Guillotina Opcional (Tratamiento Asimétrico)
    if eliminar_en_train and outliers_train > 0:
        print(f"  🔪 [ASIMETRÍA MLOPS] Eliminando {outliers_train} filas anómalas SOLO del set de Entrenamiento...")
        mascara_inliers = X_tr_trans[nombre_bandera] == 0

        X_tr_trans = X_tr_trans[mascara_inliers].reset_index(drop=True)
        if y_tr_trans is not None:
            y_tr_trans = y_tr_trans[mascara_inliers].reset_index(drop=True)

        print("    ↳ Purga completada. La matriz predictora y el target siguen perfectamente alineados.")
    else:
        print("  🚩 [ASIMETRÍA MLOPS] Conservando filas. La inteligencia del árbol usará la bandera de anomalía.")

    print("-" * 80)
    print("  🛡️ ESTATUS: Bosque de Aislamiento desplegado. Outliers bajo control estricto.")
    print("  💾 Artefacto IsolationForest guardado para la API de Producción.")
    print(f"\n⏱️ Detección de Outliers completada en {time.time() - inicio_timer:.3f}s")

    return X_tr_trans, X_te_trans, y_tr_trans, rutas, bosque

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    print(">>> 🚂 ENTRENANDO DETECTOR MULTIVARIADO EN TRAIN <<<")
    X_train_out, _, y_train_out, rutas_actualizadas, modelo_iforest = deteccion_outliers_aislamiento(
        X_train=manager.X_train, 
        y_train=manager.y_train, 
        rutas=manager.rutas,
        contamination='auto',
        eliminar_en_train=False # 🛑 Arquitecto: Cambia a True si quieres purgar las filas anómalas en Train
    )

    print("\n>>> 🔒 APLICANDO DETECTOR MULTIVARIADO A TEST <<<")
    _, X_test_out, _, _, _ = deteccion_outliers_aislamiento(
        X_train=manager.X_train, # Dummy requerido por la firma original para compatibilidad
        X_test=manager.X_test, 
        rutas=manager.rutas,
        modelo_entrenado=modelo_iforest # El Puente MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_out
    manager.X_test = X_test_out
    manager.y_train = y_train_out
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Asegurar inicialización de la caja fuerte de modelos si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['detector_outliers_iforest'] = modelo_iforest

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Detección de Outliers: {e}")


# In[110]:


X_train.info()


# In[111]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict

def winsorizacion_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    limites: Tuple[float, float] = (0.001, 0.999),
    receta_aprendida: Dict[str, Tuple[float, float]] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 4 - Paso 12.2] Motor AutoML de Winsorización (Capping).
    - Muro MLOps: Calcula los percentiles matemáticos SOLO en Train y los hereda a Test.
    - Filtro de Tipos: Solo opera sobre variables estrictamente numéricas (ignora booleanos y categorías).
    - Tolerancia a NaNs: El cálculo de percentiles y el recorte (.clip) ignoran los valores nulos.
    - Prevención de Colapso: Evita comprimir variables con varianza cero.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🗜️ FASE 12.2: Tratamiento de Outliers (Winsorización al {limites[0]*100}% - {limites[1]*100}%) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # 1. Escudo de Tipos: Solo queremos comprimir métricas reales, no banderas 0/1
    cols_numericas = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]

    if not cols_numericas:
        print("  ✅ [BYPASS] No se encontraron variables numéricas continuas para comprimir.")
        return X_trans, rutas, {}

    # ==========================================
    # 2. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if receta_aprendida is not None:
        print("  🔒 [TEST] Aplicando techos y pisos aprendidos de Train...")
        columnas_comprimidas = 0

        for col, (limite_inf, limite_sup) in receta_aprendida.items():
            if col in X_trans.columns:
                # .clip() recorta los extremos y deja los NaNs intactos
                X_trans[col] = X_trans[col].clip(lower=limite_inf, upper=limite_sup)
                columnas_comprimidas += 1

        print(f"    ↳ Replicado en {columnas_comprimidas} variables (Picos extremos recortados a ciegas).")
        print(f"\n⏱️ Winsorización completada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, receta_aprendida

    # ==========================================
    # 3. Modo TRAIN (.fit)
    # ==========================================
    print(f"  🚂 [TRAIN] Calculando percentiles para {len(cols_numericas)} variables continuas...")
    diccionario_produccion = {}
    columnas_comprimidas = 0

    for col in cols_numericas:
        # Extraemos la serie ignorando los NaNs
        serie_limpia = X_trans[col].dropna()

        if len(serie_limpia) == 0:
            continue # Si la columna es puro NaN, la ignoramos

        # Calculamos los límites matemáticos (Piso y Techo)
        limite_inf = serie_limpia.quantile(limites[0])
        limite_sup = serie_limpia.quantile(limites[1])

        # Guardamos la receta si los límites son lógicos (evita comprimir variables que son un solo número)
        if limite_inf < limite_sup:
            diccionario_produccion[col] = (limite_inf, limite_sup)

            # Aplicamos la compresión
            valores_extremos_antes = ((X_trans[col] < limite_inf) | (X_trans[col] > limite_sup)).sum()
            X_trans[col] = X_trans[col].clip(lower=limite_inf, upper=limite_sup)

            columnas_comprimidas += 1
            if valores_extremos_antes > 0:
                print(f"    🔄 '{col}': {valores_extremos_antes} valores anómalos comprimidos a [{limite_inf:.2f}, {limite_sup:.2f}]")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Topología estabilizada. {columnas_comprimidas} dimensiones comprimidas.")
    print("  💾 Diccionario de percentiles guardado para la API de Producción.")
    print(f"\n⏱️ Winsorización completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, diccionario_produccion

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # Configuramos los límites: Cortamos el 0.1% inferior y el 0.1% superior (El 99.8% de la data queda intacta)
    limites_elegidos = (0.001, 0.999)

    print(">>> 🚂 ENTRENANDO WINSORIZADOR EN TRAIN <<<")
    X_train_win, rutas_actualizadas, receta_winsor = winsorizacion_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        limites=limites_elegidos
    )

    print("\n>>> 🔒 APLICANDO WINSORIZADOR A TEST <<<")
    X_test_win, _, _ = winsorizacion_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        receta_aprendida=receta_winsor # El Puente MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_win
    manager.X_test = X_test_win
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Asegurar inicialización de la caja fuerte de modelos si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['receta_winsorizacion'] = receta_winsor

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Winsorización: {e}")


# In[112]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def escalamiento_dinamico_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    requiere_escalamiento: bool = True,
    metodo: str = 'minmax',
    escalador_entrenado: Optional[object] = None
) -> Tuple[pd.DataFrame, Dict, Optional[object]]:
    """
    [FASE 4 - Paso 13.1] Motor AutoML de Escalamiento Dinámico.
    - Switch Inteligente: Si requiere_escalamiento=False, hace bypass (ideal para ecosistemas 100% Árboles).
    - Muro MLOps: Aprende los rangos máximos/mínimos SOLO en Train. Aplica ciegamente en Test.
    - Preservación Topológica: Solo escala variables continuas (num_vars), dejando booleanas y 
      categorías nativas intactas. Retorna un DataFrame de Pandas, no un array de Numpy.
    - 🚀 Optimización de Memoria (NUEVO): Convierte el resultado float64 nativo de Sklearn a float32.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 13.1: Escalamiento Dinámico ({metodo.upper()}) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # 1. El Interruptor AutoML (Regla lrl1)
    if not requiere_escalamiento:
        print("  ✅ [BYPASS] Escalamiento omitido. La matriz está optimizada para algoritmos basados en Árboles (Scale-Invariant).")
        return X_trans, rutas, escalador_entrenado

    # 2. Escudo de Tipos: Seleccionamos SOLO las numéricas continuas
    # Las booleanas ya son 0 y 1. Las categóricas nativas son intocables.
    cols_a_escalar = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]

    if not cols_a_escalar:
        print("  ✅ [BYPASS] No hay variables numéricas continuas para escalar.")
        return X_trans, rutas, escalador_entrenado

    # Extraemos índices y columnas para reconstruir el DataFrame post-Sklearn
    indices = X_trans.index

    # ==========================================
    # 3. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if escalador_entrenado is not None:
        print(f"  🔒 [TEST] Comprimiendo {len(cols_a_escalar)} dimensiones usando la escala memorizada de Train...")

        # Transformamos y reinyectamos en el DataFrame
        matriz_escalada = escalador_entrenado.transform(X_trans[cols_a_escalar])

        # 🚀 DOWNCASTING AUTOMÁTICO: Forzamos float32 para evitar el sobrepeso de float64 de sklearn
        X_trans.loc[:, cols_a_escalar] = matriz_escalada.astype(np.float32)

        print(f"\n⏱️ Escalamiento completado en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, escalador_entrenado

    # ==========================================
    # 4. Modo TRAIN (.fit_transform)
    # ==========================================
    print(f"  🚂 [TRAIN] Entrenando escalador '{metodo}' sobre {len(cols_a_escalar)} variables continuas...")

    # Selección del Motor Matemático
    if metodo == 'minmax':
        escalador = MinMaxScaler()  # Comprime estrictamente entre 0 y 1
    elif metodo == 'standard':
        escalador = StandardScaler() # Media 0, Varianza 1
    elif metodo == 'robust':
        escalador = RobustScaler()   # Usa la mediana y el IQR (inmune a outliers extremos que sobrevivieron)
    else:
        raise ValueError(f"🛑 Método '{metodo}' no soportado. Usa 'minmax', 'standard' o 'robust'.")

    # Aprendemos los rangos matemáticos (fit) y comprimimos la matriz (transform)
    matriz_escalada = escalador.fit_transform(X_trans[cols_a_escalar])

    # 🚀 DOWNCASTING AUTOMÁTICO: Forzamos float32 para evitar el sobrepeso de float64 de sklearn
    X_trans.loc[:, cols_a_escalar] = matriz_escalada.astype(np.float32)

    print("-" * 80)
    print("  🛡️ ESTATUS: Espacio vectorial estandarizado. Listo para Deep Learning y Meta-Modelos.")
    print("  💾 Artefacto Escalador guardado para la API de Producción.")
    print(f"\n⏱️ Escalamiento completado en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, escalador


# ==========================================
# 🔧 NUEVO: Herramienta de Traducción Inversa
# ==========================================
def traductor_inverso_escalamiento(
    X_escalado: pd.DataFrame, 
    escalador_entrenado: object
) -> pd.DataFrame:
    """
    [Herramienta MLOps] Traductor de Escalamiento Inverso (Máquina del Tiempo).
    - Toma una matriz escalada (con 0s y 1s) y utiliza la memoria fotográfica del 
      escalador para devolverle sus valores lógicos del mundo real (Dólares, Años, Horas).
    """
    if escalador_entrenado is None:
        return X_escalado.copy()

    X_traducido = X_escalado.copy()

    # Scikit-learn (versiones modernas) guarda las columnas que aprendió en .feature_names_in_
    if hasattr(escalador_entrenado, 'feature_names_in_'):
        cols_memorizadas = escalador_entrenado.feature_names_in_
    else:
        raise ValueError("🛑 El escalador no tiene memoria de las columnas. Requiere Scikit-Learn reciente.")

    # Filtramos para asegurarnos de que solo intentamos traducir las columnas que existen
    cols_validas = [c for c in cols_memorizadas if c in X_traducido.columns]

    if cols_validas:
        # La magia matemática ocurre aquí (.inverse_transform)
        matriz_original = escalador_entrenado.inverse_transform(X_traducido[cols_validas])
        X_traducido.loc[:, cols_validas] = matriz_original

    return X_traducido


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # DECISIÓN DEL ARQUITECTO: ¿Usaremos Redes Neuronales / Regresión Logística después?
    VAMOS_A_USAR_DEEP_LEARNING = False 
    METODO_ESCALAMIENTO = 'minmax' 

    print(">>> 🚂 ENTRENANDO ESCALADOR EN TRAIN <<<")
    X_train_esc, rutas_actualizadas, modelo_escalador = escalamiento_dinamico_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        requiere_escalamiento=VAMOS_A_USAR_DEEP_LEARNING,
        metodo=METODO_ESCALAMIENTO
    )

    print("\n>>> 🔒 APLICANDO ESCALADOR A TEST <<<")
    X_test_esc, _, _ = escalamiento_dinamico_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        requiere_escalamiento=VAMOS_A_USAR_DEEP_LEARNING,
        escalador_entrenado=modelo_escalador # El Puente MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_esc
    manager.X_test = X_test_esc
    manager.rutas = rutas_actualizadas

    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['escalador_numerico'] = modelo_escalador

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

    # --- DEMOSTRACIÓN DEL TRADUCTOR INVERSO ---
    print("\n>>> ⏪ PRUEBA DE LA MÁQUINA DEL TIEMPO (REVERSO) <<<")
    if VAMOS_A_USAR_DEEP_LEARNING and modelo_escalador is not None:
        # Tomamos el primer paciente/registro de Test (que ahora es un conjunto de decimales)
        paciente_ejemplo = manager.X_test.head(1).copy()

        # 🔧 FIX BLINDADO: Obligamos al código a agarrar una variable puramente NUMÉRICA que fue escalada
        cols_escaladas = getattr(modelo_escalador, 'feature_names_in_', [])
        if len(cols_escaladas) > 0:
            variable_prueba = cols_escaladas[0] # Tomamos la primera variable numérica segura

            valor_escalado = paciente_ejemplo[variable_prueba].values[0]
            print(f"🤖 Visión Máquina (Escalado): {variable_prueba} = {valor_escalado:.4f}")

            # Lo pasamos por el Traductor
            paciente_traducido = traductor_inverso_escalamiento(paciente_ejemplo, modelo_escalador)
            valor_original = paciente_traducido[variable_prueba].values[0]

            # Usamos formateo dinámico: si el original es entero, no mostramos decimales
            if float(valor_original).is_integer():
                print(f"👤 Visión Humana (Original): {variable_prueba} = {int(valor_original):,}")
            else:
                print(f"👤 Visión Humana (Original): {variable_prueba} = {valor_original:,.2f}")
        else:
             print("⚠️ No hay variables numéricas escaladas para la demostración.")
    else:
        print("  ⏭️ Bypass activo o escalador no entrenado. Traducción inversa omitida.")

except Exception as e:
    print(f"🛑 Error en el Escalamiento Dinámico o Traductor: {e}")


# In[113]:


X_train.head()


# In[114]:


X_test.head()


# In[115]:


X_train.info()


# # FASE 5: Ingeniería Profunda y Aumento de Datos (La Vanguardia)
# Con una matriz limpia y numérica, desplegamos IA generativa y análisis causal.
# 
# 

# In[116]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, List, Optional

def ingenieria_rezagos_automl(
    X: pd.DataFrame, 
    rutas: Dict, 
    topologia_dataset: str = 'Transversal', # 🔧 NUEVO: Enrutador Topológico Maestro
    columnas_a_rezagar: List[str] = None,
    periodos: List[int] = [1, 2, 3],
    variable_entidad_id: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    [FASE 5 - Paso 14.1] Motor AutoML de Variables de Rezago (Lag Features).
    - Auto-Descubrimiento Temporal: Busca columnas datetime para usarlas como ancla.
    - Inteligencia Topológica: Bypass si es 'Transversal'. Actúa si es 'Serie de Tiempo Pura' o 'Datos de Panel'.
    - Exterminio del ID (NUEVO): Destruye la entidad y el index tras rezagar para evitar Leakage.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🕰️ FASE 14.1: Ingeniería de Rezagos Temporales (Lag Features) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # ==========================================
    # 1. Auditoría de Seguridad AutoML (El Bypass Inteligente)
    # ==========================================
    if topologia_dataset == 'Transversal':
        print(f"  ✅ [BYPASS INTELIGENTE] Topología detectada como '{topologia_dataset}'.")
        print("    ↳ Operación abortada para prevenir mezcla caótica de registros independientes.")
        return X_trans, rutas

    # 🚀 NUEVA INTELIGENCIA: Escáner de Topología Temporal
    columnas_fecha = X_trans.select_dtypes(include=['datetime64', 'datetime', 'datetimetz']).columns.tolist()

    if not columnas_fecha:
        print(f"  ⚠️ [ADVERTENCIA] Topología es '{topologia_dataset}', pero NO hay columnas tipo 'datetime' vivas.")
        print("    ↳ Bypass activado por seguridad.")
        return X_trans, rutas

    variable_tiempo = columnas_fecha[0]
    print(f"  🧭 [AUTO-DETECCIÓN TEMPORAL] Ancla cronológica encontrada: '{variable_tiempo}'.")

    # --- AUTO-DETECCIÓN DE VARIABLES NUMÉRICAS ---
    if not columnas_a_rezagar:
        columnas_a_rezagar = [c for c in rutas.get('num_vars', []) if c in X_trans.columns and c != variable_tiempo]
        if not columnas_a_rezagar:
            print("  ✅ [BYPASS] No se encontraron variables numéricas en el enrutador para rezagar.")
            return X_trans, rutas
        else:
            print(f"  🧠 [AUTO-DETECCIÓN VARIABLES] Se detectaron {len(columnas_a_rezagar)} variables numéricas para analizar.")

    # ==========================================
    # 2. Ordenamiento y Topología Temporal
    # ==========================================
    print(f"  🚂 [TRAIN/TEST] Generando memoria histórica de {len(periodos)} periodos para {len(columnas_a_rezagar)} variables...")

    nombres_idx_originales = X_trans.index.names
    entidad_rescatada = False

    if topologia_dataset == 'Datos de Panel' and variable_entidad_id:
        # Rescate si estaba en el Index
        if variable_entidad_id not in X_trans.columns and variable_entidad_id in nombres_idx_originales:
            X_trans = X_trans.reset_index()
            entidad_rescatada = True
            print(f"  🔓 [RESCATE] Entidad '{variable_entidad_id}' recuperada del Index para agrupar.")

        if variable_entidad_id in X_trans.columns:
            X_trans = X_trans.sort_values(by=[variable_entidad_id, variable_tiempo])
            motor_shift = X_trans.groupby(variable_entidad_id)
            print(f"    ↳ Blindaje de Identidad activo (Datos de Panel): Agrupando por '{variable_entidad_id}'.")
        else:
            print(f"    ⚠️ [CRÍTICO] La entidad '{variable_entidad_id}' no se encontró.")
            X_trans = X_trans.sort_values(by=[variable_tiempo])
            motor_shift = X_trans
            print(f"    ↳ FALLBACK: Serie de tiempo global activa (Serie Pura).")
    else:
        X_trans = X_trans.sort_values(by=[variable_tiempo])
        motor_shift = X_trans
        print(f"    ↳ Serie de tiempo global activa (Serie Pura): Ordenando cronológicamente por '{variable_tiempo}'.")

    # ==========================================
    # 3. Creación de Multi-Universos (Lags)
    # ==========================================
    columnas_creadas = 0
    nombre_lag = None 

    for col in columnas_a_rezagar:
        if col not in X_trans.columns: continue

        for p in periodos:
            nombre_lag = f"{col}_lag_{p}"
            X_trans[nombre_lag] = motor_shift[col].shift(p)
            columnas_creadas += 1

            if nombre_lag not in rutas['num_vars']:
                rutas['num_vars'].append(nombre_lag)

    huecos_generados = X_trans[nombre_lag].isna().sum() if (columnas_creadas > 0 and nombre_lag) else 0

    # ==========================================
    # 🚀 PROTOCOLO DE EXTERMINIO: Index y Columna ID
    # ==========================================
    print(f"  🧹 Ejecutando Protocolo de Limpieza Final...")
    X_trans = X_trans.reset_index(drop=True) # Destruye cualquier index personalizado

    if topologia_dataset == 'Datos de Panel' and variable_entidad_id and variable_entidad_id in X_trans.columns:
        X_trans = X_trans.drop(columns=[variable_entidad_id])
        if variable_entidad_id in rutas.get('cat_vars', []):
            rutas['cat_vars'].remove(variable_entidad_id)
        print(f"    ↳ Entidad '{variable_entidad_id}' eliminada de las columnas para evitar Data Leakage.")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Máquina del tiempo completada. {columnas_creadas} dimensiones históricas inyectadas.")
    if huecos_generados > 0:
        print(f"  ⚠️ NOTA: Se generaron {huecos_generados} NaNs naturales por falta de pasado en los primeros registros.")
        print("  💡 CONSEJO MLOps: Deberás pasar el Imputador (Paso 11.1) nuevamente sobre estos NaNs antes de entrenar.")
    print(f"\n⏱️ Rezagos generados en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # ---------------------------------------------------------
    # 🔗 AUTOWIRING MLOPS: Extracción Segura de Variables Globales/Manager
    # ---------------------------------------------------------
    # 1. Buscamos la Topología
    TOPOLOGIA_GLOBAL = 'Transversal' # Valor por defecto seguro
    if 'reporte_topologia' in globals() and isinstance(globals()['reporte_topologia'], dict):
        TOPOLOGIA_GLOBAL = globals()['reporte_topologia'].get('topologia', 'Transversal')
    elif 'reporte_topologia' in locals() and isinstance(locals()['reporte_topologia'], dict):
         TOPOLOGIA_GLOBAL = locals()['reporte_topologia'].get('topologia', 'Transversal')

    # 2. Buscamos la Entidad
    ENTIDAD_GLOBAL = None
    if 'VARIABLE_ENTIDAD_GLOBAL' in globals():
        ENTIDAD_GLOBAL = globals()['VARIABLE_ENTIDAD_GLOBAL']
    elif 'VARIABLE_ENTIDAD_GLOBAL' in locals():
        ENTIDAD_GLOBAL = locals()['VARIABLE_ENTIDAD_GLOBAL']

    print(">>> 🚂 EVALUANDO REZAGOS EN TRAIN <<<")
    X_train_lag, rutas_actualizadas = ingenieria_rezagos_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        topologia_dataset=TOPOLOGIA_GLOBAL, 
        columnas_a_rezagar=None, 
        periodos=[1, 2],
        variable_entidad_id=ENTIDAD_GLOBAL
    )

    print("\n>>> 🔒 EVALUANDO REZAGOS EN TEST <<<")
    X_test_lag, _ = ingenieria_rezagos_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        topologia_dataset=TOPOLOGIA_GLOBAL, 
        columnas_a_rezagar=None, 
        periodos=[1, 2],
        variable_entidad_id=ENTIDAD_GLOBAL  
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_lag
    manager.X_test = X_test_lag
    manager.rutas = rutas_actualizadas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la creación de Rezagos: {e}")


# In[117]:


X_train.head()


# In[118]:


X_train.info()


# In[119]:


X_test.info()


# In[120]:


y_train.info()


# In[121]:


y_test.info()


# In[122]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import re
import time
from typing import Dict

def radar_nulos_profundos_automl(df: pd.DataFrame, nombre_matriz: str = "Matriz") -> Dict[str, int]:
    """
    [HERRAMIENTA MLOps] Escáner de Nulos Ocultos (Anomalías Léxicas).
    - Inteligencia: Solo ataca columnas de texto/categorías para ahorrar CPU.
    - Desglose de Nativos (NUEVO): Clasifica inteligentemente entre NaN (Numéricos) y NaT (Fechas).
    - Motor Regex: Detecta falsos nulos ('N/A', 'unknown', '?', '-', espacios vacíos).
    - Vectorización: Usa .str.match() nativo de Pandas en C++ (Cero bucles for).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"🛑 Error: La matriz '{nombre_matriz}' está vacía o no es válida.")
        return {}

    print(f"=== 📡 INICIANDO RADAR DE ANOMALÍAS EN: {nombre_matriz} ===")
    inicio_timer = time.time()

    reporte_ocultos = {}

    # 🚀 NUEVO: Desglose inteligente de nulos nativos por Tipo de Dato
    nulos_por_columna = df.isna().sum()
    cols_con_nulos = nulos_por_columna[nulos_por_columna > 0]

    desglose_nativos = {"NaN": 0, "NaT": 0}

    for col, cantidad in cols_con_nulos.items():
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            desglose_nativos["NaT"] += cantidad
        else:
            desglose_nativos["NaN"] += cantidad

    nulos_nativos_totales = sum(desglose_nativos.values())

    # 1. El Súper-Regex del Arquitecto
    # (?i) = Case insensitive. \s* = Ignora espacios al inicio/fin. 
    patron_falsos_nulos = re.compile(r'(?i)^\s*(unknown|n/?a|null|nan|missing|none|-1|\?|-|)\s*$')

    # 2. Escudo de Tipos: Solo escaneamos textos, la matemática pura no tiene letras
    cols_texto = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    nulos_ocultos_totales = 0

    if cols_texto:
        for col in cols_texto:
            # Aislamos solo los valores que NO son nulos nativos (para no contar doble)
            serie_viva = df[col].dropna().astype(str)

            if not serie_viva.empty:
                # Aplicamos el motor Regex vectorizado
                detecciones = serie_viva.str.match(patron_falsos_nulos).sum()

                if detecciones > 0:
                    reporte_ocultos[col] = detecciones
                    nulos_ocultos_totales += detecciones

    # 3. Reporte de Inteligencia
    print("-" * 60)
    if nulos_nativos_totales == 0 and nulos_ocultos_totales == 0:
        print("  ✅ ESTATUS: Matriz 100% Pura. Cero nulos nativos, cero nulos ocultos.")
    else:
        print(f"  ⚠️ ALERTAS ENCONTRADAS:")
        print(f"    ↳ Nulos Nativos Totales: {nulos_nativos_totales:,}")

        # Desglose específico
        if desglose_nativos['NaN'] > 0:
            print(f"       - Tipo NaN (Flotantes/Texto) : {desglose_nativos['NaN']:,}")
        if desglose_nativos['NaT'] > 0:
            print(f"       - Tipo NaT (Fechas/Tiempos)  : {desglose_nativos['NaT']:,}")

        print(f"    ↳ Nulos Ocultos (Regex)  : {nulos_ocultos_totales:,}")

        if nulos_ocultos_totales > 0:
            print("  🦠 Desglose de columnas infectadas con Nulos Léxicos:")
            for col, cantidad in reporte_ocultos.items():
                print(f"     - '{col}': {cantidad:,} registros basura")

    print(f"\n⏱️ Escaneo completado en {time.time() - inicio_timer:.3f}s\n")

    return reporte_ocultos


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X_train' not in locals() or 'X_test' not in locals():
        raise EnvironmentError("No se encontraron 'X_train' o 'X_test'.")

    print(">>> 🔍 AUDITORÍA DE SEGURIDAD PRE-ENTRENAMIENTO <<<")

    # Escaneamos Train
    infecciones_train = radar_nulos_profundos_automl(X_train, nombre_matriz="X_TRAIN")

    # Escaneamos Test
    infecciones_test = radar_nulos_profundos_automl(X_test, nombre_matriz="X_TEST")

    # Lógica de reacción automática (Opcional)
    if infecciones_train or infecciones_test:
        print("💡 CONSEJO MLOps: Se detectó basura léxica. ")
        print("   Recomendación: En tu código del Imputador KNN (Fase 11.1) o en la Guillotina, ")
        print("   deberías reemplazar estos textos por np.nan usando df.replace(regex) para que el Imputador los cure.")

except Exception as e:
    print(f"🛑 Error en el Radar de Nulos: {e}")


# In[123]:


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    # 🚀 Clean Code Absoluto: Función 100% Autónoma, lee la memoria sola.
    print(">>> 🚂 ENTRENANDO IMPUTADOR KNN EN TRAIN <<<")
    X_train_knn, rutas_actualizadas, modelo_knn = imputacion_knn_automl(
        X=manager.X_train, 
        rutas=manager.rutas,
        n_vecinos=5
    )

    print("\n>>> 🔒 APLICANDO IMPUTADOR KNN A TEST <<<")
    X_test_knn, _, _ = imputacion_knn_automl(
        X=manager.X_test, 
        rutas=manager.rutas,
        n_vecinos=5,
        imputador_entrenado=modelo_knn 
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_knn
    manager.X_test = X_test_knn
    manager.rutas = rutas_actualizadas

    # FIX MLOps: Asegurar inicialización de la caja fuerte de modelos si no existe
    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['imputador_knn'] = modelo_knn

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Imputación KNN: {e}")


# In[124]:


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X_train' not in locals() or 'X_test' not in locals():
        raise EnvironmentError("No se encontraron 'X_train' o 'X_test'.")

    print(">>> 🔍 AUDITORÍA DE SEGURIDAD PRE-ENTRENAMIENTO <<<")

    # Escaneamos Train
    infecciones_train = radar_nulos_profundos_automl(X_train, nombre_matriz="X_TRAIN")

    # Escaneamos Test
    infecciones_test = radar_nulos_profundos_automl(X_test, nombre_matriz="X_TEST")

    # Lógica de reacción automática (Opcional)
    if infecciones_train or infecciones_test:
        print("💡 CONSEJO MLOps: Se detectó basura léxica. ")
        print("   Recomendación: En tu código del Imputador KNN (Fase 11.1) o en la Guillotina, ")
        print("   deberías reemplazar estos textos por np.nan usando df.replace(regex) para que el Imputador los cure.")

except Exception as e:
    print(f"🛑 Error en el Radar de Nulos: {e}")


# In[125]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import warnings
import gc # 🚀 NUEVO: Garbage Collector para RAM
from typing import Tuple, Dict, Optional
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import itertools

def generacion_guiada_llm_fe(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    es_regresion: bool = True,
    max_features_nuevas: int = 5,
    receta_formulas: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    [FASE 5 - Paso 14.2] Generador ReAct (LLM-FE) + Juez Bayesiano.
    - Sandbox Matemático: Ejecuta expresiones de forma segura.
    - Arquitectura Big Data (NUEVO): Submuestreo estricto para el Juez Bayesiano y GC para la RAM.
    - Muro MLOps: En Train descubre y aprueba fórmulas. En Test aplica estrictamente.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🧬 FASE 14.2: LLM-FE ReAct y Evaluación Bayesiana ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    cols_numericas = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]
    entorno_seguro = {"np": np, "X": X_trans}

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (Aplicador Ciego)
    # ==========================================
    if receta_formulas is not None:
        if not receta_formulas:
            print("  ✅ [BYPASS] No hay Súper-Características aprendidas de Train para inyectar.")
            return X_trans, rutas, receta_formulas

        print(f"  🔒 [TEST] Inyectando {len(receta_formulas)} Súper-Características aprendidas de Train...")
        for nombre_feature, formula in receta_formulas.items():
            try:
                X_trans[nombre_feature] = eval(formula, {"__builtins__": {}}, entorno_seguro)
            except Exception as e:
                print(f"  ⚠️ Error inyectando '{nombre_feature}': {e}. Llenando con 0.")
                X_trans[nombre_feature] = 0.0

        print(f"\n⏱️ Inyección completada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, receta_formulas

    # ==========================================
    # 2. Modo TRAIN (El Laboratorio del Agente ReAct)
    # ==========================================
    if y is None or not cols_numericas:
        print("  ✅ [BYPASS] Se requiere la variable objetivo 'y' y variables numéricas para el Juez Bayesiano.")
        return X_trans, rutas, {}

    print(f"  🚂 [TRAIN] Inicializando Tribunal Bayesiano y Agente Generador...")

    # --- A. Entrenar Modelo Base (Juez) con Arquitectura Big Data ---
    MAX_EVAL_SAMPLES = 5000 # 🚀 Blindaje RAM: Máximo de filas para evaluar combinaciones

    juez = BayesianRidge() if es_regresion else GaussianNB()

    if len(X_trans) > MAX_EVAL_SAMPLES:
        print(f"    ↳ Dataset masivo. Creando 'Submuestra de Tribunal' de {MAX_EVAL_SAMPLES:,} filas para proteger RAM...")
        # Tomamos una muestra estratificada/aleatoria rápida (usamos head para no romper series de tiempo)
        X_juez = X_trans[cols_numericas].head(MAX_EVAL_SAMPLES).fillna(0)
        y_juez = y.head(MAX_EVAL_SAMPLES)
    else:
        X_juez = X_trans[cols_numericas].fillna(0)
        y_juez = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scoring_metric = 'r2' if es_regresion else 'accuracy'
        score_base = np.mean(cross_val_score(juez, X_juez, y_juez, cv=3, scoring=scoring_metric))

    print(f"    ⚖️ Score Base (Red Bayesiana en Submuestra): {score_base:.4f}")

    # --- B. El Agente ReAct ---
    top_cols = cols_numericas[:10] 
    operadores = ['+', '-', '*'] 

    formulas_propuestas = {}
    print(f"  🧠 Agente formulando hipótesis combinatorias...")

    for col_A, col_B in itertools.combinations(top_cols, 2):
        for op in operadores:
            nombre = f"llm_{col_A}_{op}_{col_B}".replace('.','').replace('-','_')
            formula = f"X['{col_A}'] {op} X['{col_B}']"
            formulas_propuestas[nombre] = formula

    # --- C. Tribunal de Evaluación Bayesiana ---
    receta_ganadoras = {}
    mejor_score_actual = score_base

    print(f"  🧑‍⚖️ Juez evaluando {len(formulas_propuestas)} propuestas del Agente...")

    # 🚀 Entorno seguro especial para el Juez (operando solo sobre la submuestra)
    entorno_juez = {"np": np, "X": X_juez}

    for nombre_feature, formula in formulas_propuestas.items():
        if len(receta_ganadoras) >= max_features_nuevas:
            break 

        try:
            # 🚀 Blindaje RAM: X_temp se crea y se destruye en cada iteración
            X_temp = X_juez.copy()
            nueva_col_array = eval(formula, {"__builtins__": {}}, entorno_juez)
            X_temp[nombre_feature] = nueva_col_array.fillna(0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_nuevo = np.mean(cross_val_score(juez, X_temp, y_juez, cv=3, scoring=scoring_metric))

            margen_mejora = score_nuevo - mejor_score_actual
            if margen_mejora > 0.001: 
                print(f"    🌟 ¡Aprobada! [{nombre_feature}] aportó mejora (+{margen_mejora:.4f})")
                receta_ganadoras[nombre_feature] = formula
                mejor_score_actual = score_nuevo

            # 🚀 Garbage Collector: Liberar RAM explícitamente después del juicio
            del X_temp
            del nueva_col_array
            gc.collect()

        except Exception as e:
            continue

    # --- D. Inyección Definitiva en Train (Matriz Completa) ---
    if receta_ganadoras:
        print(f"  🧬 Inyectando {len(receta_ganadoras)} características evolutivas en la matriz principal...")
        for nombre, formula in receta_ganadoras.items():
            # Aquí sí inyectamos a toda la matriz X_trans original
            X_trans[nombre] = eval(formula, {"__builtins__": {}}, entorno_seguro)
            rutas['num_vars'].append(nombre)
    else:
        print("  ❌ El Juez Bayesiano rechazó todas las propuestas. Ninguna aportó valor real.")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Ingeniería LLM-FE completada con Arquitectura Big Data. Score final: {mejor_score_actual:.4f}")
    print(f"\n⏱️ Evolución terminada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, receta_ganadoras


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    TARGET_ES_REGRESION = pd.api.types.is_float_dtype(manager.y_train) or manager.y_train.nunique() > 10

    print(">>> 🚂 ENTRENANDO AGENTE LLM-FE EN TRAIN <<<")
    X_train_llm, rutas_actualizadas, diccionario_formulas = generacion_guiada_llm_fe(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        es_regresion=TARGET_ES_REGRESION,
        max_features_nuevas=5 
    )

    print("\n>>> 🔒 APLICANDO FÓRMULAS LLM-FE A TEST <<<")
    X_test_llm, _, _ = generacion_guiada_llm_fe(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas,
        receta_formulas=diccionario_formulas # El puente MLOps
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_llm
    manager.X_test = X_test_llm
    manager.rutas = rutas_actualizadas

    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['receta_llm_fe'] = diccionario_formulas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Fase LLM-FE: {e}")


# In[126]:


X_train.info()


# In[127]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
import gc
from typing import Tuple, Dict
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

def embeddings_ggpl_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    n_arboles: int = 15,
    profundidad: int = 3,
    modelo_gbdt_aprendido = None
) -> Tuple[pd.DataFrame, Dict, any]:
    """
    [FASE 5 - Paso 15.1] Motor AutoML de Embeddings GGPL (GBDT Leaf Encoding).
    - Proyección Dimensional: Usa un GBDT ligero para discretizar continuas.
    - Inteligencia de Tarea: Detecta si 'y' es continua o categórica.
    - Protección RAM (NUEVO): Submuestreo para el entrenamiento del GBDT y GC explícito.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🌳 FASE 15.1: Embeddings GGPL (Proyección GBDT Ligero) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}

    # Escudo de Tipos: El GBDT solo necesita las variables numéricas
    cols_numericas = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]

    if not cols_numericas:
        print("  ✅ [BYPASS] No hay variables numéricas para proyectar.")
        return X_trans, rutas, modelo_gbdt_aprendido

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (.apply)
    # ==========================================
    if modelo_gbdt_aprendido is not None:
        print(f"  🔒 [TEST] Pasando matriz por el GBDT aprendido para extraer coordenadas de hojas...")

        X_num_sana = X_trans[cols_numericas].fillna(0)
        hojas = modelo_gbdt_aprendido.apply(X_num_sana)

        hojas_planas = hojas.reshape(hojas.shape[0], -1)

        n_features_nuevas = hojas_planas.shape[1]
        nombres_nuevas = [f"gbdt_emb_{i}" for i in range(n_features_nuevas)]

        X_trans[nombres_nuevas] = hojas_planas

        # 🧹 Limpieza
        del X_num_sana
        del hojas
        del hojas_planas
        gc.collect()

        print(f"    ↳ {n_features_nuevas} nuevas coordenadas proyectadas y mapeadas.")
        print(f"\n⏱️ Proyección GGPL completada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, modelo_gbdt_aprendido

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if y is None:
        print("  ✅ [BYPASS] Se requiere la variable objetivo 'y' para entrenar el GBDT Supervisor.")
        return X_trans, rutas, None

    es_regresion = pd.api.types.is_float_dtype(y) or y.nunique() > 10

    # ⚙️ Parámetros de Escalabilidad RAM
    MAX_GBDT_SAMPLES = 5000 

    X_num_sana = X_trans[cols_numericas].fillna(0)

    # --- PROTECCIÓN RAM: Submuestreo solo para el fit ---
    if len(X_num_sana) > MAX_GBDT_SAMPLES:
        print(f"  🚂 [TRAIN] Dataset masivo. Entrenando GBDT en submuestra de {MAX_GBDT_SAMPLES} filas...")
        X_fit = X_num_sana.sample(n=MAX_GBDT_SAMPLES, random_state=42)
        y_fit = y.loc[X_fit.index]
    else:
        print(f"  🚂 [TRAIN] Entrenando GBDT en dataset completo...")
        X_fit = X_num_sana
        y_fit = y

    if es_regresion:
        gbdt = GradientBoostingRegressor(n_estimators=n_arboles, max_depth=profundidad, random_state=42)
    else:
        gbdt = GradientBoostingClassifier(n_estimators=n_arboles, max_depth=profundidad, random_state=42)

    print(f"    ↳ Entrenando red de {n_arboles} árboles (Profundidad: {profundidad})...")
    gbdt.fit(X_fit, y_fit)

    # El apply() se hace a toda la matriz X_num_sana para no perder registros
    print(f"    ↳ Extrayendo hiper-coordenadas (Leaf Indices) de toda la matriz...")
    hojas = gbdt.apply(X_num_sana)
    hojas_planas = hojas.reshape(hojas.shape[0], -1)

    n_features_nuevas = hojas_planas.shape[1]
    nombres_nuevas = [f"gbdt_emb_{i}" for i in range(n_features_nuevas)]

    X_trans[nombres_nuevas] = hojas_planas
    rutas['cat_vars'].extend(nombres_nuevas)

    # 🧹 Limpieza agresiva de memoria
    del X_num_sana
    del X_fit
    del y_fit
    del hojas
    del hojas_planas
    gc.collect()

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Discretización GGPL Exitosa. {n_features_nuevas} variables categóricas de alta densidad creadas.")
    print("  🧠 Las variables continuas ahora tienen un gemelo no lineal.")
    print(f"\n⏱️ Proyección GGPL completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, gbdt

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    print(">>> 🚂 ENTRENANDO EMBEDDINGS GGPL EN TRAIN <<<")
    X_train_emb, rutas_actualizadas, modelo_gbdt_embedder = embeddings_ggpl_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        n_arboles=15, 
        profundidad=3
    )

    print("\n>>> 🔒 PROYECTANDO EMBEDDINGS GGPL EN TEST <<<")
    X_test_emb, _, _ = embeddings_ggpl_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas,
        modelo_gbdt_aprendido=modelo_gbdt_embedder 
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_emb
    manager.X_test = X_test_emb
    manager.rutas = rutas_actualizadas

    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['embedder_gbdt'] = modelo_gbdt_embedder

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en la Fase de Embeddings GGPL: {e}")


# In[128]:


X_train.info()


# In[129]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import warnings
import gc
from typing import Tuple, Dict, List, Optional
try:
    import shap
    import lightgbm as lgb
except ImportError:
    print("🛑 MLOps Warning: Las librerías 'shap' y 'lightgbm' son requeridas para esta fase.")
    print("   Ejecuta: !pip install shap lightgbm")

def sinergias_shap_diamond_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    max_sinergias: int = 5,
    receta_sinergias: Optional[List[Tuple[str, str]]] = None
) -> Tuple[pd.DataFrame, Dict, Optional[List[Tuple[str, str]]]]:
    """
    [FASE 5 - Paso 15.2] Motor AutoML de Sinergias (SHAP Interaction 2D + Diamond FDR).
    - Optimización O(M^2): Filtra el Top 10 de variables antes de calcular la matriz SHAP.
    - Framework Diamond (Knockoffs): Crea variables "sombra" para controlar el False Discovery Rate (FDR).
    - Protección RAM Estricta: Límite estricto de 5k filas para SHAP y liberación explícita de memoria (GC).
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 💎 FASE 15.2: Sinergias Causales (SHAP 2D + Diamond FDR Framework) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': []}
    cols_numericas = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]

    if len(cols_numericas) < 2:
        print("  ✅ [BYPASS] Se necesitan al menos 2 variables numéricas para buscar sinergias.")
        return X_trans, rutas, receta_sinergias

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if receta_sinergias is not None:
        if not receta_sinergias:
            print("  ✅ [BYPASS] Train no descubrió sinergias significativas. Matriz intacta.")
            return X_trans, rutas, receta_sinergias

        print(f"  🔒 [TEST] Inyectando {len(receta_sinergias)} sinergias exactas descubiertas en Train...")
        for col_A, col_B in receta_sinergias:
            if col_A in X_trans.columns and col_B in X_trans.columns:
                nombre_sinergia = f"sinergia_{col_A}_X_{col_B}"
                X_trans[nombre_sinergia] = X_trans[col_A].astype(float) * X_trans[col_B].astype(float)

        print(f"\n⏱️ Inyección de sinergias completada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, receta_sinergias

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if y is None:
        print("  ✅ [BYPASS] Se requiere la variable objetivo 'y' para calcular SHAP Interactions.")
        return X_trans, rutas, None

    es_regresion = pd.api.types.is_float_dtype(y) or y.nunique() > 10

    # ⚙️ Parámetros de Escalabilidad RAM (SHAP 2D es hiper-pesado)
    MAX_SHAP_SAMPLES = 5000 

    # --- PASO A: Filtro de Élite (Prevención de Explosión de RAM) ---
    print(f"  🕵️‍♂️ Entrenando modelo de reconocimiento rápido (LGBM) - Límite: {MAX_SHAP_SAMPLES} filas...")
    X_num = X_trans[cols_numericas].fillna(0)

    # Submuestreo estricto para proteger RAM durante cálculos matemáticos complejos
    if len(X_num) > MAX_SHAP_SAMPLES:
        X_sample = X_num.sample(n=MAX_SHAP_SAMPLES, random_state=42)
    else:
        X_sample = X_num

    y_sample = y.loc[X_sample.index]

    if es_regresion:
        modelo_base = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    else:
        modelo_base = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)

    modelo_base.fit(X_sample, y_sample)

    # Extraemos el Top 10 para no hacer un SHAP cruzado gigante
    importancias = pd.Series(modelo_base.feature_importances_, index=cols_numericas)
    top_10_cols = importancias.nlargest(10).index.tolist()

    # 🧹 Limpieza de memoria intermedia
    del modelo_base
    gc.collect()

    if len(top_10_cols) < 2:
        return X_trans, rutas, []

    # --- PASO B: SHAP Interaction Values (2D) ---
    print(f"  🌌 Calculando Hiperespacio SHAP 2D para el Top {len(top_10_cols)} de variables...")
    X_top = X_sample[top_10_cols]

    # Volvemos a entrenar solo con el Top 10 para el Explainer
    modelo_shap = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1) if es_regresion else lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    modelo_shap.fit(X_top, y_sample)

    explainer = shap.TreeExplainer(modelo_shap)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # interaction_values shape: (n_samples, n_features, n_features)
        shap_interactions = explainer.shap_interaction_values(X_top)

    # 🧹 Limpieza de modelos pesados
    del modelo_shap
    del explainer
    gc.collect()

    # 🚀 FIX: Manejo robusto del array de interacciones SHAP
    # Si es una lista (suele pasar en clasificación multiclase con versiones viejas de SHAP)
    if isinstance(shap_interactions, list):
        # Tomamos la clase positiva (índice 1) si es binario, o la primera si hay más
        idx_clase = 1 if len(shap_interactions) > 1 else 0
        matriz_base = shap_interactions[idx_clase]
    # Si es un numpy array, validamos sus dimensiones
    elif isinstance(shap_interactions, np.ndarray):
        if len(shap_interactions.shape) == 4:
            # Shape (n_samples, n_features, n_features, n_classes) -> Promediamos las clases o tomamos la clase 1
            matriz_base = shap_interactions[:, :, :, 1] if shap_interactions.shape[3] > 1 else shap_interactions[:, :, :, 0]
        else:
             # Shape estándar (n_samples, n_features, n_features)
            matriz_base = shap_interactions
    else:
        # Fallback de seguridad
        matriz_base = np.array(shap_interactions)

    # Matriz simétrica de importancia absoluta media
    interaccion_media = np.abs(matriz_base).mean(axis=0)

    # Extraemos los pares con mayor interacción (ignorando la diagonal que son los efectos principales)
    candidatos = []
    for i in range(len(top_10_cols)):
        for j in range(i + 1, len(top_10_cols)):
            candidatos.append((interaccion_media[i, j], top_10_cols[i], top_10_cols[j]))

    candidatos.sort(reverse=True, key=lambda x: x[0]) 
    top_candidatos = candidatos[:max_sinergias * 2] 

    # --- PASO C: El Tribunal Diamond (Knockoffs / Control FDR) ---
    print(f"  ⚖️ Iniciando Tribunal Diamond (Control de FDR) para {len(top_candidatos)} candidatos...")

    sinergias_aprobadas = []

    for fuerza_shap, col_A, col_B in top_candidatos:
        if len(sinergias_aprobadas) >= max_sinergias: break

        sombra_B = X_sample[col_B].sample(frac=1, random_state=42).values

        interaccion_real = X_sample[col_A] * X_sample[col_B]
        interaccion_sombra = X_sample[col_A] * sombra_B

        df_torneo = pd.DataFrame({'Real': interaccion_real, 'Sombra': interaccion_sombra})
        modelo_juez = lgb.LGBMRegressor(n_estimators=20, random_state=42, verbose=-1) if es_regresion else lgb.LGBMClassifier(n_estimators=20, random_state=42, verbose=-1)
        modelo_juez.fit(df_torneo, y_sample)

        importancia_real = modelo_juez.feature_importances_[0]
        importancia_sombra = modelo_juez.feature_importances_[1]

        if importancia_real > (importancia_sombra * 1.5): 
            print(f"    🌟 [FDR Pass] Sinergia real: ({col_A} × {col_B}) > Ruido Sombra.")
            sinergias_aprobadas.append((col_A, col_B))

            nombre_sinergia = f"sinergia_{col_A}_X_{col_B}"
            X_trans[nombre_sinergia] = X_trans[col_A].astype(float) * X_trans[col_B].astype(float)
            rutas['num_vars'].append(nombre_sinergia)
        else:
            print(f"    ❌ [FDR Drop] Falso Descubrimiento detectado: ({col_A} × {col_B}) no superó a sombra.")

        # 🧹 Limpieza por cada ciclo del torneo
        del df_torneo
        del modelo_juez
        gc.collect()

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Descubrimiento Causal completado. {len(sinergias_aprobadas)} sinergias inyectadas.")
    print(f"\n⏱️ Fase SHAP+Diamond terminada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, sinergias_aprobadas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 CALCULANDO SHAP 2D Y KNOCKOFFS EN TRAIN <<<")
    X_train_shap, rutas_actualizadas, receta_interacciones = sinergias_shap_diamond_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        max_sinergias=5
    )

    print("\n>>> 🔒 APLICANDO SINERGIAS EXACTAS EN TEST <<<")
    X_test_shap, _, _ = sinergias_shap_diamond_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas,
        receta_sinergias=receta_interacciones 
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_shap
    manager.X_test = X_test_shap
    manager.rutas = rutas_actualizadas

    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['receta_sinergias_shap'] = receta_interacciones

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en Sinergias SHAP/Diamond: {e}")


# In[130]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
import warnings
from typing import Tuple, Dict
try:
    import lightgbm as lgb
except ImportError:
    print("🛑 MLOps Warning: La librería 'lightgbm' es requerida para esta fase.")
    print("   Ejecuta: !pip install lightgbm")

def contrafactuales_sensibilidad_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    modelo_oraculo = None
) -> Tuple[pd.DataFrame, Dict, any]:
    """
    [FASE 5 - Paso 15.3] Motor AutoML de Contrafactuales de Sensibilidad.
    - Proxy Causal: Usa un Oráculo (LGBM) para medir la distancia a la frontera de decisión.
    - Clean Code: Extrae 'Margen de Frontera' y 'Fuerza Logit' sin intervención manual.
    - Inteligencia de Tarea: Adaptable a Clasificación (Binaria/Multiclase) y Regresión.
    - Protección RAM: Submuestreo estricto para el Oráculo y Garbage Collection.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== 🧲 FASE 15.3: Contrafactuales de Sensibilidad (Distancia a la Frontera) ===")
    inicio_timer = time.time()

    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    # 🛡️ Solo pasamos variables numéricas al Oráculo para evitar crashes categóricos
    cols_numericas = [c for c in rutas.get('num_vars', []) if c in X_trans.columns]

    if not cols_numericas:
        print("  ✅ [BYPASS] No hay variables numéricas para calcular sensibilidad espacial.")
        return X_trans, rutas, modelo_oraculo

    # Función interna para calcular las métricas espaciales
    def inyectar_distancias(df: pd.DataFrame, predicciones: np.ndarray, es_regresion: bool) -> Tuple[pd.DataFrame, list]:
        df_out = df.copy()
        nuevas_cols = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if es_regresion:
                # En regresión, la "distancia" es qué tan lejos está de la mediana global del Oráculo
                mediana_global = np.median(predicciones)
                df_out['cf_distancia_mediana'] = np.abs(predicciones - mediana_global)
                df_out['cf_desviacion_relativa'] = (predicciones - mediana_global) / (np.abs(mediana_global) + 1e-6)
                nuevas_cols = ['cf_distancia_mediana', 'cf_desviacion_relativa']
                rutas['num_vars'].extend(nuevas_cols)
            else:
                # En clasificación binaria o multiclase
                if len(predicciones.shape) == 1 or predicciones.shape[1] == 1: # Binario
                    prob_positiva = predicciones if len(predicciones.shape) == 1 else predicciones[:, 0]
                    # Distancia absoluta a la duda (0.5)
                    df_out['cf_distancia_frontera'] = np.abs(prob_positiva - 0.5)
                    # Log-Odds (Fuerza de empuje) con clip para evitar log(0)
                    p_clip = np.clip(prob_positiva, 1e-5, 1 - 1e-5)
                    df_out['cf_fuerza_logit'] = np.log(p_clip / (1 - p_clip))
                    nuevas_cols = ['cf_distancia_frontera', 'cf_fuerza_logit']
                    rutas['num_vars'].extend(nuevas_cols)
                else: # Multiclase
                    # Distancia entre la clase más probable y la segunda más probable (Margen de Confianza)
                    prob_ordenada = np.sort(predicciones, axis=1)
                    df_out['cf_margen_multiclase'] = prob_ordenada[:, -1] - prob_ordenada[:, -2]
                    df_out['cf_entropia_decision'] = -np.sum(predicciones * np.log(np.clip(predicciones, 1e-5, 1)), axis=1)
                    nuevas_cols = ['cf_margen_multiclase', 'cf_entropia_decision']
                    rutas['num_vars'].extend(nuevas_cols)
        return df_out, nuevas_cols

    # ==========================================
    # 1. Modo TEST / PRODUCCIÓN (.transform)
    # ==========================================
    if modelo_oraculo is not None:
        print(f"  🔒 [TEST] Consultando al Oráculo para medir sensibilidad de nuevos registros...")

        es_regresion = isinstance(modelo_oraculo, lgb.LGBMRegressor)
        X_num_sana = X_trans[cols_numericas].fillna(0)

        # El oráculo predice probabilidades (clasificación) o valores (regresión)
        if es_regresion:
            predicciones = modelo_oraculo.predict(X_num_sana)
        else:
            predicciones = modelo_oraculo.predict_proba(X_num_sana)
            if predicciones.shape[1] == 2: predicciones = predicciones[:, 1] # Binario

        X_trans, columnas_creadas = inyectar_distancias(X_trans, predicciones, es_regresion)

        # 🧹 Limpieza
        del X_num_sana
        gc.collect()

        print(f"    ↳ Coordenadas inyectadas: {columnas_creadas}")
        print(f"\n⏱️ Sensibilidad calculada en {time.time() - inicio_timer:.3f}s")
        return X_trans, rutas, modelo_oraculo

    # ==========================================
    # 2. Modo TRAIN (.fit)
    # ==========================================
    if y is None:
        print("  ✅ [BYPASS] Se requiere la variable objetivo 'y' para entrenar el Oráculo.")
        return X_trans, rutas, None

    es_regresion = pd.api.types.is_float_dtype(y) or y.nunique() > 10

    # ⚙️ Parámetros de Escalabilidad RAM (El Oráculo no necesita todos los datos para entender el espacio)
    MAX_ORACLE_SAMPLES = 20000 

    X_num_sana = X_trans[cols_numericas].fillna(0)

    if len(X_num_sana) > MAX_ORACLE_SAMPLES:
        print(f"  🚂 [TRAIN] Dataset masivo. Entrenando Oráculo en submuestra de {MAX_ORACLE_SAMPLES} filas...")
        X_fit = X_num_sana.sample(n=MAX_ORACLE_SAMPLES, random_state=42)
        y_fit = y.loc[X_fit.index]
    else:
        print(f"  🚂 [TRAIN] Entrenando Oráculo Espacial...")
        X_fit = X_num_sana
        y_fit = y

    # Entrenamos el Oráculo
    if es_regresion:
        oraculo = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    else:
        oraculo = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)

    oraculo.fit(X_fit, y_fit)

    print(f"    ↳ Mapeando distancias a la frontera de decisión para toda la matriz...")
    if es_regresion:
        predicciones = oraculo.predict(X_num_sana)
    else:
        predicciones = oraculo.predict_proba(X_num_sana)
        if predicciones.shape[1] == 2: predicciones = predicciones[:, 1]

    X_trans, columnas_creadas = inyectar_distancias(X_trans, predicciones, es_regresion)

    # 🧹 Purificación de RAM
    rutas['num_vars'] = list(set(rutas['num_vars'])) # Eliminar duplicados en las rutas
    del X_num_sana
    del X_fit
    del y_fit
    gc.collect()

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Inyección Contrafactual completada. La matriz ahora conoce su propia vulnerabilidad.")
    print(f"  🌟 Nuevas variables creadas: {columnas_creadas}")
    print(f"\n⏱️ Sensibilidad calculada en {time.time() - inicio_timer:.3f}s")

    return X_trans, rutas, oraculo

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 MIDIENDO DISTANCIAS CONTRAFACTUALES EN TRAIN <<<")
    X_train_cf, rutas_actualizadas, modelo_oraculo = contrafactuales_sensibilidad_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas
    )

    print("\n>>> 🔒 PROYECTANDO DISTANCIAS CONTRAFACTUALES EN TEST <<<")
    X_test_cf, _, _ = contrafactuales_sensibilidad_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas,
        modelo_oraculo=modelo_oraculo 
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_cf
    manager.X_test = X_test_cf
    manager.rutas = rutas_actualizadas

    if not hasattr(manager, 'modelos_preprocesamiento'):
        manager.modelos_preprocesamiento = {}

    manager.modelos_preprocesamiento['oraculo_contrafactual'] = modelo_oraculo

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en Contrafactuales de Sensibilidad: {e}")


# In[131]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, Optional

def configuracion_equidad_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    umbral_desbalance: float = 0.8
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    [FASE 5 - Paso 16.1] Radar AutoML: Diagnóstico Universal de Equidad.
    - RAM Shield: NO clona ni inventa filas (Bye SMOTE). Deja la matriz intacta.
    - Inteligencia Dual: 
        ↳ Binario -> Calcula Asymmetric Bagging.
        ↳ Multiclase -> Calcula Pesos Suavizados (Smoothed Weights) anti-sobreconfianza.
    - Escudo de Producción: Si es modo TEST (y=None), pasa en milisegundos.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 16.1: Radar Universal de Equidad MLOps ===")
    inicio_timer = time.time()

    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    # ==========================================
    # 1. ESCUDOS MLOPS (Producción)
    # ==========================================
    if y is None:
        print("  ✅ [BYPASS ESTRATÉGICO] Modo TEST/Producción detectado. Matriz protegida.")
        return X.copy(), None, rutas

    # ==========================================
    # 2. Modo TRAIN (Diagnóstico de Tarea)
    # ==========================================
    es_regresion = pd.api.types.is_float_dtype(y) and y.nunique() > 20

    if es_regresion:
        print("  ✅ [BYPASS] Tarea de Regresión detectada. Las técnicas de equidad de clases se omiten.")
        return X.copy(), y.copy(), rutas

    # ==========================================
    # 3. Radar de Desbalance (El Cerebro Matemático)
    # ==========================================
    conteo_clases = y.value_counts()
    es_multiclase = len(conteo_clases) > 2

    # ----------------------------------------------------
    # MOTOR A: MULTICLASE (Pesos Suavizados Anti-Mentiras)
    # ----------------------------------------------------
    if es_multiclase:
        print(f"  🎯 [INFO] Target Multiclase detectado ({len(conteo_clases)} categorías).")
        print("     ↳ LightGBM no soporta Asymmetric Bagging nativo en Multiclase.")
        print("     ↳ Activando Plan B: Cost-Sensitive Learning (Pesos Suavizados por Raíz Cuadrada)...")

        clase_mayor = conteo_clases.max()
        clase_menor = conteo_clases.min()
        ratio_peor = clase_menor / clase_mayor

        if ratio_peor >= umbral_desbalance:
            print("  ✅ [BYPASS] Las clases están suficientemente equilibradas. Sin intervención requerida.")
            rutas['asymmetric_bagging'] = {}
        else:
            # Cálculo matemático de suavizado: sqrt(Mayor / Actual)
            # Esto ayuda a la minoría sin disparar las probabilidades al 99% como hace 'balanced'
            pesos_suavizados = {}
            for clase, count in conteo_clases.items():
                peso = np.sqrt(clase_mayor / count)
                pesos_suavizados[clase] = round(peso, 4)

            config_equidad = {'class_weight': pesos_suavizados}
            rutas['asymmetric_bagging'] = config_equidad

            print(f"  🛡️ ESTATUS: Pesos de clase estabilizados calculados: {pesos_suavizados}")
            print("     ↳ Configuración inyectada en 'rutas' para la Fase 18.")

    # ----------------------------------------------------
    # MOTOR B: BINARIO (Asymmetric Bagging Nativo)
    # ----------------------------------------------------
    else:
        clase_negativa_count = conteo_clases.get(0, 1) # Mayoritaria (pobres)
        clase_positiva_count = conteo_clases.get(1, 1) # Minoritaria (ricos)

        ratio_desbalance = clase_positiva_count / clase_negativa_count

        print(f"  📊 Diagnóstico de Equidad Binaria: Ratio Positiva/Negativa = {ratio_desbalance:.3f} (Umbral: {umbral_desbalance})")
        print(f"     ↳ Distribución original: \n{conteo_clases.to_string()}")

        if ratio_desbalance >= umbral_desbalance:
            print("  ✅ [BYPASS] Las clases están suficientemente equilibradas. Sin intervención requerida.")
            rutas['asymmetric_bagging'] = {}
        else:
            print(f"  🧬 [TRAIN] Desbalance severo detectado. Trazando plano para Asymmetric Bagging...")

            fraccion_negativa_optima = min(ratio_desbalance * 1.1, 1.0)

            config_equidad = {
                'pos_bagging_fraction': 1.0,
                'neg_bagging_fraction': round(fraccion_negativa_optima, 4),
                'bagging_freq': 1,      
                'bagging_seed': 42
            }

            rutas['asymmetric_bagging'] = config_equidad
            print("  🛡️ ESTATUS: Estrategia calculada con éxito. La matriz se mantiene intacta.")
            print(f"     ↳ Configuración inyectada en 'rutas' para la Fase 18: {config_equidad}")

    print(f"\n⏱️ Radar Universal completado en {time.time() - inicio_timer:.3f}s")

    return X.copy(), y.copy(), rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    print(">>> 🚂 CALCULANDO ESTRATEGIA UNIVERSAL DE EQUIDAD EN TRAIN (VÍA MANAGER) <<<")

    # Ejecutamos consumiendo los datos directamente del manager
    X_train_limpio, y_train_limpio, rutas_actualizadas = configuracion_equidad_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas
    )

    # Guardamos los resultados
    manager.X_train = X_train_limpio
    manager.y_train = y_train_limpio
    manager.rutas = rutas_actualizadas

    print("\n>>> 🔒 PASANDO TEST POR EL ESCUDO DE EQUIDAD <<<")
    X_test_limpio, _, _ = configuracion_equidad_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )
    manager.X_test = X_test_limpio 

    # Variables globales de transición
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en el Pipeline (Fase de Equidad): {e}")


# In[132]:


# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.y_train is None:
        raise ValueError("El Manager no tiene cargado el vector 'y_train'.")

    # AUTO-DETECCIÓN DEL NOMBRE DEL TARGET (De los pasos anteriores, o por defecto)
    nombre_target_heredado = manager.y_train.name if manager.y_train.name else "Target_Manager"

    # Ejecutamos la radiografía usando el target almacenado en el manager
    clase_minoritaria_global = diagnosticar_balance_target(
        y=manager.y_train, 
        nombre_target=nombre_target_heredado 
    )

    if clase_minoritaria_global is not None:
        # Guardamos la clase minoritaria en la memoria del manager (rutas) para usarla en el futuro
        manager.rutas['clase_minoritaria'] = clase_minoritaria_global
        print(f"📦 [MLOps] Clase minoritaria '{clase_minoritaria_global}' capturada en el Manager y lista para Fase 4.3.")

except Exception as e:
    print(f"🛑 Error en el Diagnóstico: {e}")


# In[133]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
from typing import Tuple, Dict, Optional

try:
    import lightgbm as lgb
except ImportError:
    print("🛑 MLOps Warning: La librería 'lightgbm' es requerida para el Oráculo Preliminar.")
    print("   Ejecuta: !pip install lightgbm")

def pseudo_labeling_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    X_unlabeled: Optional[pd.DataFrame] = None,
    rutas: Dict = None,
    umbral_confianza: float = 0.99
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict]:
    """
    [FASE 5 - Paso 16.2] Motor AutoML de Pseudo-Labeling (>99% Confianza).
    - Candado de Ejecución Única: Evita que el usuario corra la celda dos veces y contamine los Folds.
    - Muro MLOps Absoluto: Si es modo TEST (y=None), pasa la matriz intacta. NUNCA se contamina Test.
    - Semi-Supervisado: Aprovecha datos sin etiqueta (X_unlabeled) para enriquecer Train.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz base (X) está vacía.")

    print(f"=== 🏷️ FASE 16.2: Pseudo-Labeling y Enriquecimiento Semi-Supervisado ===")
    inicio_timer = time.time()

    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    # ==========================================
    # 1. ESCUDOS MLOPS (Producción y Doble Ejecución)
    # ==========================================
    # Escudo 1: Protección de Producción / Test
    if y is None:
        print("  ✅ [BYPASS ESTRATÉGICO] Modo TEST/Producción detectado. Matriz protegida.")
        return X.copy(), None, rutas

    # Escudo 2: Candado de Ejecución Única
    if rutas.get('pseudo_labeling_ejecutado', False):
        print("  ✅ [ESCUDO ACTIVO] Pseudo-Labeling ya fue ejecutado previamente. Bloqueando doble ejecución.")
        return X.copy(), y.copy(), rutas

    X_trans = X.copy()

    # ==========================================
    # 2. Diagnóstico de Viabilidad
    # ==========================================
    y_trans = y.copy()
    es_regresion = pd.api.types.is_float_dtype(y_trans) and y_trans.nunique() > 20

    if es_regresion:
        print("  ✅ [BYPASS] Tarea de Regresión detectada. Pseudo-Labeling requiere probabilidades de clase.")
        return X_trans, y_trans, rutas

    if X_unlabeled is None or X_unlabeled.empty:
        print("  ✅ [BYPASS] No se proporcionó matriz de datos sin etiquetar (X_unlabeled). Operación omitida.")
        return X_trans, y_trans, rutas

    # ==========================================
    # 3. Entrenamiento del Oráculo Preliminar Fuerte
    # ==========================================
    print(f"  🧠 [TRAIN] Entrenando Oráculo Preliminar para evaluar {len(X_unlabeled):,} registros oscuros...")

    # 🛡️ Filtramos solo las variables numéricas que existen en ambas matrices
    cols_numericas = [col for col in X_trans.select_dtypes(include=[np.number]).columns 
                    if col in X_unlabeled.columns]

    X_num = X_trans[cols_numericas].fillna(0)
    X_unl_num = X_unlabeled[cols_numericas].fillna(0)

    oraculo = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    oraculo.fit(X_num, y_trans)

    # ==========================================
    # 4. Inquisición de Confianza (>99%)
    # ==========================================
    print(f"  🔍 Escaneando probabilidades en la matriz sin etiqueta (Umbral: {umbral_confianza*100}%)...")
    probabilidades = oraculo.predict_proba(X_unl_num)

    # Obtenemos la confianza máxima para cada registro y la clase a la que pertenece
    max_probs = np.max(probabilidades, axis=1)
    clases_predichas = np.argmax(probabilidades, axis=1)

    # 🛡️ EL ESCUDO ANTI-VENENO: Solo los que superan el 99%
    mascara_elite = max_probs >= umbral_confianza
    candidatos_aprobados = np.sum(mascara_elite)

    if candidatos_aprobados == 0:
        print(f"  ❌ [RECHAZO] Ningún registro oscuro alcanzó el {umbral_confianza*100}% de certeza. Matriz protegida.")
    else:
        print(f"  🌟 [APROBADO] Se encontraron {candidatos_aprobados:,} registros con certeza absoluta. Inyectando...")

        # Extraemos los registros de élite de la matriz original sin etiquetar (con todas sus columnas)
        X_elite = X_unlabeled[mascara_elite].copy()

        # Extraemos las clases predichas de élite (mapeando de vuelta si las clases no son 0, 1, 2...)
        clases_reales = oraculo.classes_
        y_elite = pd.Series(clases_reales[clases_predichas[mascara_elite]], index=X_elite.index)

        # Fusionamos con la matriz principal de Train
        X_trans = pd.concat([X_trans, X_elite], axis=0, ignore_index=True)
        y_trans = pd.concat([y_trans, y_elite], axis=0, ignore_index=True)

    # 🧹 Purga de RAM
    del X_num
    del X_unl_num
    del oraculo
    del probabilidades
    gc.collect()

    # 🔒 Activamos el Candado para el futuro
    rutas['pseudo_labeling_ejecutado'] = True

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Pseudo-Labeling finalizado. Tamaño actual de Train: {len(X_trans):,} registros.")
    print(f"\n⏱️ Operación completada en {time.time() - inicio_timer:.3f}s")

    return X_trans, y_trans, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    # Simulación: Si tienes un dataset aparte sin etiquetas, lo pasas aquí.
    # X_datos_sin_etiqueta = pd.read_csv('datos_oscuros.csv')
    X_datos_sin_etiqueta = None

    print(">>> 🚂 EJECUTANDO PSEUDO-LABELING EN TRAIN <<<")
    # Limpio, automático y blindado por el Arquitecto
    X_train_semi, y_train_semi, rutas_actualizadas = pseudo_labeling_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        X_unlabeled=X_datos_sin_etiqueta, 
        rutas=manager.rutas,
        umbral_confianza=0.99
    )

    print("\n>>> 🔒 PASANDO TEST POR EL ESCUDO DE PSEUDO-LABELING <<<")
    X_test_semi, _, _ = pseudo_labeling_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_semi
    manager.y_train = y_train_semi
    manager.X_test = X_test_semi
    manager.rutas = rutas_actualizadas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en Pseudo-Labeling: {e}")


# In[134]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from typing import Tuple, Dict

# ==========================================
# MOTOR UNIFICADO: REWEIGHING IPF + AUDITORÍA VISUAL
# ==========================================
def aplicar_y_auditar_equidad_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    umbral_dir: float = 0.80 
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    [FASE 5 - Paso 16.3] Motor AutoML Unificado de Justicia Algorítmica.
    - Detección Inteligente: Diccionario Exhaustivo + Regex estricto + Escudo Anti-Ruido.
    - Curación (IPF): Reweighing Interseccional Dinámico Multidimensional.
    - Validación: Graficador automático post-tratamiento de TODAS las variables sensibles.
    - Muro MLOps: Bypass automático para Test.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    inicio_timer = time.time()
    X_trans = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}
    pesos_instancia = pd.Series(1.0, index=X_trans.index, name="sample_weight")

    # ==========================================
    # 1. ESCUDOS Y BYPASS DE MLOPS
    # ==========================================
    if y is None:
        print("  🔒 [TEST] Bypass activado. Retornando pesos neutrales (1.0) sin alterar ni graficar.")
        return X_trans, pesos_instancia

    es_regresion = pd.api.types.is_float_dtype(y) and y.nunique() > 20
    if es_regresion:
        print("  ✅ [BYPASS] Tarea de Regresión Continua. Reweighing omitido.")
        return X_trans, pesos_instancia

    print(f"=== ⚖️ FASE 16.3: Reweighing Quirúrgico (IPF) y Auditoría Ponderada ===")

    conteo_clases = y.value_counts(normalize=True)
    clase_favorable = conteo_clases.index[-1]

    # ==========================================
    # 2. RADAR DE DETECCIÓN (EL ESCUDO DEL ARQUITECTO)
    # ==========================================
    def construir_patron(palabra):
        return fr'(^{palabra}$|^{palabra}_|_{palabra}$|_{palabra}_|[a-z]{palabra.capitalize()})'

    terminos_legales = [
        # 1. EDAD Y NACIMIENTO (Age & Birth)
        'age', 'edad', 'dob', 'dateofbirth', 'birth', 'birthdate', 'birthyear', 'nacimiento', 
        'fechanacimiento', 'anonacimiento', 'year', 'año', 'ano', 'generation', 'generacion',
        # 2. SEXO, GÉNERO Y ORIENTACIÓN (Sex, Gender & Orientation)
        'sex', 'sexo', 'gender', 'genero', 'female', 'femenino', 'male', 'masculino', 
        'mujer', 'hombre', 'orientation', 'orientacion', 'sexuality', 'sexualidad', 
        'sexualorientation', 'orientacionsexual', 'lgbt', 'lgbtq', 'trans', 'transgender', 
        'transgenero', 'nonbinary', 'nobinario', 'intersex', 'intersexual',
        # 3. RAZA, ETNIA Y ORIGEN (Race, Ethnicity & Origins)
        'race', 'raza', 'ethnic', 'etnia', 'ethnicity', 'ethniccode', 'codigoetnico',
        'color', 'origin', 'origen', 'ancestry', 'ascendencia', 'minority', 'minoria', 
        'indigenous', 'indigena', 'tribe', 'tribu', 'hispanic', 'hispano', 'latino', 
        'afro', 'afroamerican', 'black', 'negro', 'white', 'blanco', 'asian', 'asiatico', 
        'caucasian', 'caucasico',
        # 4. RELIGIÓN Y CREENCIAS (Religion & Beliefs)
        'religion', 'belief', 'creencia', 'faith', 'fe', 'creed', 'credo', 'worship', 'culto',
        'muslim', 'musulman', 'jewish', 'judio', 'christian', 'cristiano', 'catholic', 
        'catolico', 'islam', 'judaismo', 'cristianismo',
        # 5. NACIONALIDAD E INMIGRACIÓN (Nationality & Immigration)
        'national', 'nacional', 'nationality', 'nacionalidad', 'nation', 'nacion', 
        'country', 'pais', 'citizen', 'ciudadano', 'citizenship', 'ciudadania', 
        'immigrant', 'inmigrante', 'immigration', 'inmigracion', 'migrant', 'migrante', 
        'refugee', 'refugiado', 'asylum', 'asilo', 'alien', 'extranjero', 'native', 'nativo',
        # 6. SALUD, DISCAPACIDAD Y GENÉTICA (Health, Disability & Genetics)
        'health', 'salud', 'medical', 'medico', 'disability', 'discapacidad', 'handicap', 
        'minusvalia', 'disabled', 'discapacitado', 'disease', 'enfermedad', 'illness', 
        'condition', 'condicion', 'genetic', 'genetico', 'pregnant', 'embarazada', 
        'pregnancy', 'embarazo', 'maternity', 'maternidad', 'paternity', 'paternidad',
        # 7. ESTADO CIVIL Y FAMILIA (Marital Status & Family)
        'marital', 'conyugal', 'maritalstatus', 'estadocivil', 'civilstatus', 'civil', 
        'marriage', 'matrimonio', 'wedding', 'spouse', 'esposo', 'esposa', 'conyuge', 
        'widow', 'viudo', 'viuda', 'divorced', 'divorciado', 'single', 'soltero', 
        'family', 'familia', 'children', 'hijos', 'dependent', 'dependents', 
        'dependiente', 'dependientes',
        # 8. SOCIOECONÓMICO Y EDUCACIÓN (Socioeconomic & Education)
        'income', 'ingreso', 'ingresos', 'salary', 'salario', 'wage', 'sueldo', 'wealth', 
        'riqueza', 'poverty', 'pobreza', 'class', 'clase', 'estrato', 'socioeconomic', 
        'socioeconomico', 'education', 'educacion', 'degree', 'grado', 'school', 'escuela', 
        'university', 'universidad', 'illiterate', 'analfabeto',
        # 9. SISTEMA PENAL Y CUSTODIA (Legal & Custody Status)
        'legalstatus', 'estadolegal', 'custodystatus', 'estadocustodia', 'custody', 'custodia', 
        'felon', 'felony', 'conviction', 'condena', 'antecedente', 'parole', 'probation',
        # 10. IDIOMA Y POLÍTICA (Language, Politics & Unions)
        'language', 'idioma', 'lenguaje', 'tongue', 'lengua', 'dialect', 'dialecto',
        'politics', 'politica', 'political', 'politico', 'union', 'tradeunion', 'sindicato', 'gremio'
    ]

    patrones_completos = [construir_patron(t) for t in terminos_legales]
    patron_sensible = re.compile('|'.join(patrones_completos), re.IGNORECASE)

    columnas_sensibles = []
    for col in X_trans.columns:
        if patron_sensible.search(col):
            col_lower = col.lower()
            # ESCUDO ANTI-RUIDO
            if 'is_missing_' in col_lower or 'missing_' in col_lower: continue
            if col_lower.startswith('cf_'): continue  # 👈 Bloqueo estricto de Contrafactuales Oráculo
            if pd.api.types.is_datetime64_any_dtype(X_trans[col]) or col in rutas.get('date_vars', []): continue
            if re.search(r'(_sin$|_cos$)', col_lower): continue
            columnas_sensibles.append(col)

    if not columnas_sensibles:
        print("  ✅ [INFO] No se detectaron columnas protegidas útiles.")
        return X_trans, pesos_instancia

    # ==========================================
    # 3. MOTOR IPF (ITERATIVE PROPORTIONAL FITTING)
    # ==========================================
    df_calc = pd.DataFrame({'Target_Binario': (y == clase_favorable).astype(int)})

    for col in columnas_sensibles:
        if pd.api.types.is_numeric_dtype(X_trans[col]) and X_trans[col].nunique() > 10:
            df_calc[f'S_{col}'] = pd.qcut(X_trans[col], q=4, duplicates='drop').astype(str)
        else:
            df_calc[f'S_{col}'] = X_trans[col].astype(str)

    cols_a_corregir = []
    for col in columnas_sensibles:
        tasas = df_calc.groupby(f'S_{col}')['Target_Binario'].mean()
        if tasas.max() > 0 and (tasas / tasas.max()).min() < umbral_dir:
            cols_a_corregir.append(col)

    if cols_a_corregir:
        EPOCHS = 5 
        for epoch in range(EPOCHS):
            for col in cols_a_corregir:
                s_col = f'S_{col}'
                df_calc['peso'] = pesos_instancia
                peso_total = df_calc['peso'].sum()

                p_y = df_calc.groupby('Target_Binario')['peso'].sum() / peso_total
                p_s = df_calc.groupby(s_col)['peso'].sum() / peso_total
                p_s_y = df_calc.groupby([s_col, 'Target_Binario'])['peso'].sum() / peso_total

                for (s_val, y_val), p_sy_val in p_s_y.items():
                    if p_sy_val > 0:
                        w_update = (p_s[s_val] * p_y[y_val]) / p_sy_val
                        mask = (df_calc[s_col] == s_val) & (df_calc['Target_Binario'] == y_val)
                        pesos_instancia.loc[mask] *= w_update

        pesos_instancia = pesos_instancia * (len(pesos_instancia) / pesos_instancia.sum())
        print(f"  🚨 ESTATUS: Matriz curada IPF. Se neutralizaron {len(cols_a_corregir)} variables con sesgo.")
    else:
        print("  ✅ ESTATUS: Ninguna variable requirió intervención IPF.")

    # ==========================================
    # 4. AUDITORÍA VISUAL POST-TRATAMIENTO
    # ==========================================
    print("\n  📊 GENERANDO REPORTE DE VALIDACIÓN DE EQUIDAD PONDERADA...")
    df_calc['peso_instancia'] = pesos_instancia
    sns.set_theme(style="whitegrid")

    # MODIFICACIÓN: Iteramos sobre TODAS las columnas sensibles detectadas, no solo las corregidas.
    for col in columnas_sensibles: 
        col_analisis = f'S_{col}'
        display(Markdown(f"### 🔍 Verificación Post-Tratamiento en: `{col}`"))

        # CÁLCULO PONDERADO ESTRICTO
        def calc_weighted_metrics(g):
            peso_total_grupo = g['peso_instancia'].sum()
            exitos_ponderados = (g['Target_Binario'] * g['peso_instancia']).sum()
            tasa = exitos_ponderados / peso_total_grupo if peso_total_grupo > 0 else 0
            return pd.Series({'Tasa_Exito': tasa, 'Muestra_Total': len(g)})

        tabla_tasas = df_calc.groupby(col_analisis).apply(calc_weighted_metrics).reset_index()
        tabla_tasas.sort_values(by='Tasa_Exito', ascending=False, inplace=True)

        if tabla_tasas.empty: continue

        grupo_privilegiado = tabla_tasas.iloc[0][col_analisis]
        tasa_maxima = tabla_tasas.iloc[0]['Tasa_Exito']

        if tasa_maxima == 0:
            tabla_tasas['DIR'] = 1.0
        else:
            tabla_tasas['DIR'] = tabla_tasas['Tasa_Exito'] / tasa_maxima

        plt.figure(figsize=(10, 4))
        ax = sns.barplot(data=tabla_tasas, x=col_analisis, y='Tasa_Exito', palette='crest')
        plt.axhline(tasa_maxima * 0.8, color='red', linestyle='--', label='Límite Legal (80%)')
        plt.title(f"Tasa EQUILIBRADA de obtención de '{clase_favorable}' por {col}", fontsize=14)
        plt.ylabel("Probabilidad de Éxito (Ponderada)")
        plt.ylim(0, max(0.5, tasa_maxima + 0.1))
        plt.xticks(rotation=15) 
        plt.legend()
        plt.show()

        print(f"  👑 Base de Nivelación: '{grupo_privilegiado}' (Tasa ponderada: {tasa_maxima*100:.1f}%)")
        print("  " + "="*85)

        mensajes = [] 
        for _, row in tabla_tasas.iterrows():
            grupo_actual = row[col_analisis]
            dir_actual = row['DIR']
            if grupo_actual == grupo_privilegiado: continue

            if dir_actual < 0.80:
                mensajes.append(f"🚨 [ALERTA] '{grupo_actual}': DIR = {dir_actual:.2f} ({row['Tasa_Exito']*100:.1f}%)")
            else:
                mensajes.append(f"✅ [JUSTO] '{grupo_actual}': DIR = {dir_actual:.2f} ({row['Tasa_Exito']*100:.1f}%)")

        lote_size = 20
        for i in range(0, len(mensajes), lote_size):
            lote = mensajes[i:i + lote_size]
            mitad = (len(lote) + 1) // 2 
            for j in range(mitad):
                col1 = lote[j]
                col2 = lote[j + mitad] if (j + mitad) < len(lote) else ""
                print(f"  {col1:<40} |  {col2}")
            if (i + lote_size) < len(mensajes): print("  " + "-"*85)

        print("  " + "="*85 + "\n")

    print(f"⏱️ Pipeline unificado completado en {time.time() - inicio_timer:.3f}s")
    return X_trans, pesos_instancia

# ==========================================
# Celda de Ejecución Maestra en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    print(">>> 🚂 EJECUTANDO PIPELINE UNIFICADO DE EQUIDAD EN TRAIN <<<")
    # Calcula pesos Y grafica al mismo tiempo usando IPF
    X_train_ipf, pesos_train = aplicar_y_auditar_equidad_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        umbral_dir=0.80 
    )

    print("\n>>> 🔒 GENERANDO PESOS PARA TEST (ESCUDO MLOPS) <<<")
    # Genera los pesos 1.0 y silencia las gráficas
    X_test_ipf, pesos_test = aplicar_y_auditar_equidad_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_ipf
    manager.X_test = X_test_ipf
    manager.pesos_train = pesos_train

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en el Pipeline de Equidad: {e}")


# In[135]:


X_train.info()


# In[136]:


X_test.info()


# In[137]:


y_train.info()


# In[138]:


y_test.info()


# # FASE 6: Selección de Variables (El Tribunal Supremo)
# Destruyendo la redundancia creada en la Fase 5.

# In[139]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
import warnings
from typing import Tuple, Dict

def filtro_fugas_y_fechas_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    umbral_fuga: float = 0.98 # Correlación > 98% = Guillotina inmediata
) -> Tuple[pd.DataFrame, Dict]:
    """
    [FASE 6 - Paso 17.1] Tribunal Supremo: Purga de Fechas y Fugas del Futuro.
    - Purga Temporal: Elimina variables listadas en 'date_vars' Y auto-detecta columnas datetime.
    - Escáner de Fugas: Calcula la correlación de Pearson con el Target para detectar trampas.
    - Muro MLOps: Registra las fugas en Train y ejecuta la misma guillotina exacta en Test.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 17.1: El Tribunal Supremo (Target Leakage & Date Purge) ===")
    inicio_timer = time.time()

    X_clean = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}
    columnas_eliminadas = []

    # ==========================================
    # 1. PURGA DE FECHAS (Diccionario + Auto-Detección)
    # ==========================================
    # 🚀 NUEVO: Leemos el diccionario y escaneamos activamente la matriz
    fechas_registradas = rutas.get('date_vars', [])
    fechas_detectadas = X_clean.select_dtypes(include=['datetime64', 'datetime', 'datetimetz']).columns.tolist()

    # Unificamos ambas listas (sin duplicados)
    todas_las_fechas = list(set(fechas_registradas + fechas_detectadas))
    fechas_presentes = [col for col in todas_las_fechas if col in X_clean.columns]

    if fechas_presentes:
        X_clean.drop(columns=fechas_presentes, inplace=True)
        columnas_eliminadas.extend(fechas_presentes)

        # 💡 Actualizamos el diccionario para que el Manager (y Test) no las olvide
        rutas['date_vars'] = todas_las_fechas

        print(f"  📅 Purga de Fechas: Decapitadas {len(fechas_presentes)} variables temporales.")
        print(f"     ↳ {fechas_presentes}")
    else:
        print("  ✅ Purga de Fechas: No se encontraron variables base tipo Date en la matriz.")

    # ==========================================
    # 2. MODO TRAIN: Detección de Target Leakage
    # ==========================================
    if y is not None:
        print(f"  🔍 Escaneando matriz en busca de Fugas del Futuro (Correlación > {umbral_fuga*100}%)...")

        # Solo verificamos variables numéricas para el Leakage
        cols_numericas = X_clean.select_dtypes(include=[np.number]).columns

        if len(cols_numericas) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Calculamos correlación lineal absoluta contra el Target
                correlaciones = X_clean[cols_numericas].corrwith(y).abs()

            fugas_detectadas = correlaciones[correlaciones >= umbral_fuga].index.tolist()

            if fugas_detectadas:
                print(f"  🚨 [ALERTA DE FUGA] Se detectaron {len(fugas_detectadas)} variables sospechosamente perfectas:")
                print(f"     ↳ {fugas_detectadas}")
                X_clean.drop(columns=fugas_detectadas, inplace=True)
                columnas_eliminadas.extend(fugas_detectadas)

                # 💡 GUARDADO ESTRATÉGICO: Inyectamos la sentencia en el diccionario de rutas
                rutas['fugas_del_futuro'] = fugas_detectadas
            else:
                print("  ✅ Escudo Anti-Fugas: No se detectaron variables tramposas.")
                rutas['fugas_del_futuro'] = []

    # ==========================================
    # 3. MODO TEST: Ejecución de Sentencias
    # ==========================================
    else:
        fugas_heredadas = rutas.get('fugas_del_futuro', [])
        fugas_presentes = [col for col in fugas_heredadas if col in X_clean.columns]

        if fugas_presentes:
            X_clean.drop(columns=fugas_presentes, inplace=True)
            columnas_eliminadas.extend(fugas_presentes)
            print(f"  🔒 [TEST] Aplicando guillotina heredada de Train: {len(fugas_presentes)} fugas eliminadas.")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Purificación completada. Total columnas actuales: {X_clean.shape[1]} (-{len(columnas_eliminadas)} eliminadas).")
    print(f"\n⏱️ Operación completada en {time.time() - inicio_timer:.3f}s")

    # 🧹 Purga estricta de RAM
    gc.collect()

    return X_clean, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 PURGANDO FECHAS Y FUGAS EN TRAIN <<<")
    X_train_clean, rutas_actualizadas = filtro_fugas_y_fechas_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        umbral_fuga=0.98
    )

    print("\n>>> 🔒 APLICANDO GUILLOTINA HEREDADA EN TEST <<<")
    X_test_clean, _ = filtro_fugas_y_fechas_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_clean
    manager.X_test = X_test_clean
    manager.rutas = rutas_actualizadas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

except Exception as e:
    print(f"🛑 Error en el Filtro de Fugas: {e}")


# In[140]:


# ==========================================
# Celda de Ejecución en tu .ipynb
# ==========================================
try:
    if 'X_train' not in locals() or 'X_test' not in locals():
        raise EnvironmentError("No se encontraron 'X_train' o 'X_test'.")

    print(">>> 🔍 AUDITORÍA DE SEGURIDAD PRE-ENTRENAMIENTO <<<")

    # Escaneamos Train
    infecciones_train = radar_nulos_profundos_automl(X_train, nombre_matriz="X_TRAIN")

    # Escaneamos Test
    infecciones_test = radar_nulos_profundos_automl(X_test, nombre_matriz="X_TEST")

    # Lógica de reacción automática (Opcional)
    if infecciones_train or infecciones_test:
        print("💡 CONSEJO MLOps: Se detectó basura léxica. ")
        print("   Recomendación: En tu código del Imputador KNN (Fase 11.1) o en la Guillotina, ")
        print("   deberías reemplazar estos textos por np.nan usando df.replace(regex) para que el Imputador los cure.")

except Exception as e:
    print(f"🛑 Error en el Radar de Nulos: {e}")


# In[141]:


X_train.info()


# In[142]:


X_test.info()


# In[143]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
import warnings
from typing import Tuple, Dict

def guillotina_colinealidad_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    umbral_corr: float = 0.98 # Guillotina para correlaciones > 98%
) -> Tuple[pd.DataFrame, Dict]:
    """
    [FASE 6 - Paso 17.2] Tribunal Supremo: Guillotina de Colinealidad (Spearman).
    - Caza de Gemelos: Usa Spearman para detectar redundancia no lineal perfecta.
    - RAM Shield: Procesamiento optimizado de la matriz triangular superior.
    - Muro MLOps: Evalúa y condena en Train. Ejecuta la misma sentencia en Test.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 17.2: Guillotina de Colinealidad (Spearman > {umbral_corr*100}%) ===")
    inicio_timer = time.time()

    X_clean = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    # ==========================================
    # 1. MODO TRAIN: El Juicio (Cálculo de Matriz)
    # ==========================================
    if y is not None:
        print(f"  🔍 Escaneando redundancia matemática profunda en {X_clean.shape[1]} columnas...")

        # Spearman solo opera sobre números. Aislamos las numéricas en silencio.
        cols_numericas = X_clean.select_dtypes(include=[np.number]).columns.tolist()

        if len(cols_numericas) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 1.1 Cálculo de la matriz de correlación absoluta
                print("     ↳ Construyendo matriz de correlación de Spearman (Esto puede tomar unos segundos)...")
                matriz_corr = X_clean[cols_numericas].corr(method='spearman').abs()

                # 1.2 Extracción de la diagonal superior (Evita comparar A con A, o A con B y B con A)
                upper_tri = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))

                # 1.3 Identificación de los clones condenados
                columnas_a_eliminar = [col for col in upper_tri.columns if any(upper_tri[col] > umbral_corr)]

            # 🧹 Purga inmediata de RAM
            del matriz_corr
            del upper_tri
            gc.collect()

            if columnas_a_eliminar:
                print(f"  🚨 [SENTENCIA] Se detectaron {len(columnas_a_eliminar)} variables clonadas/redundantes.")
                # print(f"     ↳ {columnas_a_eliminar}") # Descomentar si deseas ver la lista completa

                X_clean.drop(columns=columnas_a_eliminar, inplace=True)

                # 💡 GUARDADO ESTRATÉGICO: Anotamos la sentencia en el registro
                rutas['gemelos_colineales'] = columnas_a_eliminar
            else:
                print("  ✅ [JUSTO] La matriz es matemáticamente pura. No hay colinealidad extrema.")
                rutas['gemelos_colineales'] = []
        else:
            print("  ⚠️ Insuficientes variables numéricas para calcular colinealidad.")
            rutas['gemelos_colineales'] = []

    # ==========================================
    # 2. MODO TEST: La Ejecución (Bypass)
    # ==========================================
    else:
        clones_heredados = rutas.get('gemelos_colineales', [])
        clones_presentes = [col for col in clones_heredados if col in X_clean.columns]

        if clones_presentes:
            X_clean.drop(columns=clones_presentes, inplace=True)
            print(f"  🔒 [TEST] Guillotina aplicada. {len(clones_presentes)} clones decapitados según reglas de Train.")
        else:
            print("  ✅ [TEST] Matriz validada. Sin clones que eliminar.")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Matriz libre de redundancia extrema. Columnas finales: {X_clean.shape[1]}")
    print(f"\n⏱️ Operación completada en {time.time() - inicio_timer:.3f}s")

    return X_clean, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    print(">>> 🚂 CAZANDO CLONES MATEMÁTICOS EN TRAIN <<<")
    X_train_clean, rutas_actualizadas = guillotina_colinealidad_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas,
        umbral_corr=0.98 # ⚖️ Tolerancia máxima de similitud
    )

    print("\n>>> 🔒 DECAPITANDO CLONES EN TEST <<<")
    X_test_clean, _ = guillotina_colinealidad_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )

    # Guardamos los activos en el Manager
    manager.X_train = X_train_clean
    manager.X_test = X_test_clean
    manager.rutas = rutas_actualizadas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

    gc.collect()

except Exception as e:
    print(f"🛑 Error en la Guillotina de Colinealidad: {e}")


# In[144]:


X_train.info()


# In[145]:


X_test.info()


# In[146]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
import re
import warnings
from typing import Tuple, Dict

try:
    import lightgbm as lgb
    import shap
except ImportError:
    print("🛑 MLOps Warning: Las librerías 'lightgbm' y 'shap' son requeridas para el Tribunal Final.")
    print("   Ejecuta: !pip install lightgbm shap")

def tribunal_boruta_shap_automl(
    X: pd.DataFrame, 
    y: pd.Series = None, 
    rutas: Dict = None,
    muestras_shap: int = 5000 # Optimización AutoML: Usar 5k registros para calcular SHAP a la velocidad de la luz
) -> Tuple[pd.DataFrame, Dict]:
    """
    [FASE 6 - Paso 17.3] El Juez Final: Boruta-SHAP & Null Importance.
    - ⚖️ FIX MLOps: Inyecta la estrategia universal de equidad (Bagging o Pesos) calculada en Fase 16.1.
    - Cirugía Anti-Leakage: Elimina 'Caballos de Troya' (cf_, gbdt_, prob_) antes del juicio.
    - Shadow Features: Crea copias permutadas (ruido puro) para establecer la línea base.
    - SHAP Values: Tensor 3D nativo que soporta tanto evaluación Binaria como Multiclase.
    - La Guillotina: Elimina variables aportando menos que el percentil 90 del ruido.
    """
    if X is None or X.empty:
        raise ValueError("🛑 Error Crítico: La matriz (X) está vacía.")

    print(f"=== ⚖️ FASE 17.3: Tribunal Boruta-SHAP (Filtro contra el Ruido) ===")
    inicio_timer = time.time()

    X_clean = X.copy()
    rutas = rutas or {'num_vars': [], 'cat_vars': [], 'bool_vars': [], 'date_vars': []}

    # ==========================================
    # 1. MODO TRAIN: El Torneo contra las Sombras
    # ==========================================
    if y is not None:
        # 🚀 FIX MLOps: Cirugía Quirúrgica contra el Target Leakage
        patron_troya = re.compile(r'(^cf_|^gbdt_|_prob_|^prob_)', re.IGNORECASE)
        meta_features = [col for col in X_clean.columns if patron_troya.search(col)]

        if meta_features:
            print(f"  🔪 [CIRUGÍA] Extirpando {len(meta_features)} 'Caballos de Troya' (Meta-Características con Leakage)...")
            X_clean.drop(columns=meta_features, inplace=True)
            rutas['meta_features_purgadas'] = meta_features
        else:
            rutas['meta_features_purgadas'] = []

        print(f"  🧠 Preparando el Torneo Predictivo para {X_clean.shape[1]} variables reales...")

        cols_numericas = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols_numericas) == 0:
            print("  ✅ [BYPASS] No hay variables numéricas para evaluar. Operación omitida.")
            rutas['basura_boruta'] = []
            return X_clean, rutas

        # 1.1 Muestreo Inteligente para Velocidad MLOps (AutoML)
        n_muestras = min(len(X_clean), muestras_shap)
        idx_sample = np.random.choice(X_clean.index, n_muestras, replace=False)
        X_sample = X_clean.loc[idx_sample, cols_numericas].fillna(0)
        y_sample = y.loc[idx_sample]

        # 1.2 Creación de los "Clones de Sombra" (Shadow Features)
        print("     ↳ Generando Clones de Sombra (Ruido Aleatorio)...")
        X_shadow = X_sample.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)

        shadow_cols = [f"shadow_{c}" for c in X_sample.columns]
        X_shadow.columns = shadow_cols

        # 1.3 La Arena de Batalla (Matriz Combinada)
        X_torneo = pd.concat([X_sample, X_shadow], axis=1)

        # 1.4 Entrenamiento Rápido del Juez (LightGBM)
        print("     ↳ Entrenando Oráculo Juez (Con Inteligencia de Equidad)...")
        es_regresion = pd.api.types.is_float_dtype(y_sample) and y_sample.nunique() > 20

        # ⚖️ ROOT FIX MLOps: Extraemos la estrategia universal de la Fase 16.1
        config_equidad = rutas.get('asymmetric_bagging', {})
        param_juez = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Inteligencia UI: El Juez reporta qué estrategia está usando
        if config_equidad:
            param_juez.update(config_equidad)
            if 'pos_bagging_fraction' in config_equidad:
                print("       ↳ (Estrategia Binaria: Asymmetric Bagging inyectado al Juez)")
            elif 'class_weight' in config_equidad:
                print("       ↳ (Estrategia Multiclase: Pesos Suavizados inyectados al Juez)")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if es_regresion:
                juez = lgb.LGBMRegressor(**param_juez)
            else:
                juez = lgb.LGBMClassifier(**param_juez)

            juez.fit(X_torneo, y_sample)

        # 1.5 Cálculo de Magnitudes Shapley (SHAP)
        print("     ↳ Calculando Magnitudes SHAP para emitir sentencias...")
        explainer = shap.TreeExplainer(juez)
        shap_values = explainer.shap_values(X_torneo)

        # 🚀 FIX MLOps: Blindaje Tridimensional para SHAP Moderno y Multiclase
        if isinstance(shap_values, list):
            shap_imp = np.zeros(X_torneo.shape[1])
            for class_vals in shap_values:
                shap_imp += np.abs(class_vals).mean(axis=0)
        else:
            if len(shap_values.shape) == 3:
                shap_imp = np.abs(shap_values).mean(axis=0).sum(axis=1)
            else:
                shap_imp = np.abs(shap_values).mean(axis=0)

        # 1.6 Veredicto
        imp_reales = pd.Series(shap_imp[:len(cols_numericas)], index=cols_numericas)
        imp_sombras = pd.Series(shap_imp[len(cols_numericas):], index=shadow_cols)

        # ⚖️ FIX MLOps: Usamos el percentil 90 del ruido
        umbral_basura = np.percentile(imp_sombras, 90)

        columnas_a_eliminar = imp_reales[imp_reales <= umbral_basura].index.tolist()

        # 🧹 Purga de RAM
        del X_sample, X_shadow, X_torneo, juez, explainer, shap_values
        gc.collect()

        if columnas_a_eliminar:
            print(f"  🚨 [SENTENCIA] A la Guillotina: {len(columnas_a_eliminar)} variables reales aportaban menos que el ruido puro.")
            X_clean.drop(columns=columnas_a_eliminar, inplace=True)
            rutas['basura_boruta'] = columnas_a_eliminar
        else:
            print("  ✅ [JUSTO] Todas las variables sobrevivieron. Son estadísticamente superiores al ruido.")
            rutas['basura_boruta'] = []

    # ==========================================
    # 2. MODO TEST: La Ejecución Silenciosa
    # ==========================================
    else:
        # 2.1 Ejecutar la Cirugía de los Caballos de Troya
        meta_heredadas = rutas.get('meta_features_purgadas', [])
        meta_presentes = [col for col in meta_heredadas if col in X_clean.columns]

        if meta_presentes:
            X_clean.drop(columns=meta_presentes, inplace=True)
            print(f"  🔒 [TEST] Cirugía aplicada. {len(meta_presentes)} 'Caballos de Troya' eliminados.")

        # 2.2 Ejecutar la Guillotina de Boruta
        basura_heredada = rutas.get('basura_boruta', [])
        basura_presente = [col for col in basura_heredada if col in X_clean.columns]

        if basura_presente:
            X_clean.drop(columns=basura_presente, inplace=True)
            print(f"  🔒 [TEST] Guillotina aplicada. {len(basura_presente)} variables decapitadas.")
        else:
            print("  ✅ [TEST] Matriz evaluada. Sin variables inútiles que purgar.")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Matriz Real Libre de Ruido. Columnas finales (Élite Predictiva): {X_clean.shape[1]}")
    print(f"\n⏱️ Tribunal Boruta-SHAP completado en {time.time() - inicio_timer:.3f}s")

    return X_clean, rutas

# ==========================================
# Celda de Ejecución en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None or manager.X_test is None:
        raise ValueError("El Manager no tiene cargadas las matrices 'X_train', 'y_train' o 'X_test'.")

    if not manager.rutas:
        raise ValueError("El Manager no tiene el ruteo de variables cargado en 'rutas'.")

    print(">>> 🚂 JUZGANDO VARIABLES CONTRA EL RUIDO EN TRAIN (BORUTA-SHAP) <<<")
    X_train_elite, rutas_actualizadas = tribunal_boruta_shap_automl(
        X=manager.X_train, 
        y=manager.y_train, 
        rutas=manager.rutas
    )

    print("\n>>> 🔒 EJECUTANDO SENTENCIA EN TEST <<<")
    X_test_elite, _ = tribunal_boruta_shap_automl(
        X=manager.X_test, 
        y=None, 
        rutas=manager.rutas
    )

    # Guardamos los activos finales en el Manager
    manager.X_train = X_train_elite
    manager.X_test = X_test_elite
    manager.rutas = rutas_actualizadas

    # (Transición) Reflejamos temporalmente en globales si el código viejo las requiere
    X_train = manager.X_train
    y_train = manager.y_train
    X_test = manager.X_test
    rutas_variables = manager.rutas

    gc.collect()

except Exception as e:
    print(f"🛑 Error en Tribunal Boruta-SHAP: {e}")


# In[147]:


X_train.info()


# In[148]:


X_test.info()


# # FASE 7: MLOps: Exportación y Calibración
# El modelo sale del laboratorio al mundo real.

# In[149]:


# ==========================================
# 0. Blindaje de Dependencias y Estética
# ==========================================
import os
import time
import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional, Any

try:
    import pyarrow as pa
    MOTOR_PARQUET = f"PyArrow v{pa.__version__}"
except ImportError:
    MOTOR_PARQUET = "Motor No Detectado"
    print("🛑 MLOps Warning: El motor binario 'pyarrow' es requerido para escribir archivos Parquet.")

# ==========================================
# 🔧 NUEVO: Traductor Universal de Tipos Numpy -> JSON
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """ Escudo de Serialización: Convierte tipos Numpy/Pandas a Python nativo """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# ==========================================
# MOTOR MLOPS: EXPORTACIÓN BINARIA PURA (V2 - CON ESCUDOS Y ARTEFACTOS)
# ==========================================
def exportar_activos_mlops_integral(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: Optional[pd.Series], 
    rutas: Dict,
    modelos_preprocesamiento: Dict[str, Any], # 🧠 NUEVO: Diccionario de cerebros del Manager
    pesos_equidad: Optional[pd.Series] = None, 
    grupos_validacion: Optional[np.ndarray] = None, 
    directorio_salida: str = "mlops_activos"
) -> None:
    """
    [FASE 7 - Paso 18.1] Exportación Serializada Integral.
    - Guarda X, y, Metadatos.
    - Serializa los pesos de justicia algorítmica (sample_weights).
    - Serializa el mapa topológico de Folds (grupos_cv).
    - 🧠 NUEVO: Serializa TODOS los cerebros de preprocesamiento (KNN, Escala, GBDT, etc.) con Joblib.
    """
    print(f"=== 💾 FASE 18.1: Exportación Binaria Integral ({MOTOR_PARQUET}) ===")
    inicio_timer = time.time()

    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    dir_artefactos = os.path.join(directorio_salida, "artefactos_preprocesamiento")
    if not os.path.exists(dir_artefactos):
        os.makedirs(dir_artefactos)

    try:
        # 1. Matrices y Vectores Base
        X_train.to_parquet(os.path.join(directorio_salida, "X_train_opt.parquet"), engine='pyarrow', index=False)
        X_test.to_parquet(os.path.join(directorio_salida, "X_test_opt.parquet"), engine='pyarrow', index=False)
        y_train.to_frame(name='Target').to_parquet(os.path.join(directorio_salida, "y_train_opt.parquet"), engine='pyarrow', index=False)
        print("  ✅ Matrices X e y exportadas exitosamente a Parquet.")

        if y_test is not None:
            y_test.to_frame(name='Target').to_parquet(os.path.join(directorio_salida, "y_test_opt.parquet"), engine='pyarrow', index=False)

        # 2. Escudo de Equidad Algorítmica (Pesos)
        if pesos_equidad is not None:
            df_pesos = pd.DataFrame({'sample_weight': pesos_equidad}).reset_index(drop=True)
            df_pesos.to_parquet(os.path.join(directorio_salida, "pesos_train.parquet"), engine='pyarrow', index=False)
            print("  ⚖️ Escudo de Equidad (Pesos) blindado en Parquet.")

        # 3. Mapa de Validación Cruzada (Grupos)
        if grupos_validacion is not None:
            df_grupos = pd.DataFrame({'grupos_cv': grupos_validacion}).reset_index(drop=True)
            df_grupos.to_parquet(os.path.join(directorio_salida, "grupos_cv.parquet"), engine='pyarrow', index=False)
            print("  🗺️ Mapa de Validación Cruzada blindado en Parquet.")

        # 4. 🧠 Serialización de Cerebros (Imputadores, Escaladores, Encoders)
        if modelos_preprocesamiento:
            print(f"  🧠 Detectados {len(modelos_preprocesamiento)} artefactos de preprocesamiento. Serializando...")
            for nombre_artefacto, objeto_artefacto in modelos_preprocesamiento.items():
                if objeto_artefacto is not None:
                    ruta_artefacto = os.path.join(dir_artefactos, f"{nombre_artefacto}.joblib")
                    joblib.dump(objeto_artefacto, ruta_artefacto)
            print("  📦 Todos los cerebros de preprocesamiento han sido congelados criogénicamente (Joblib).")

        # 5. Diccionario de Rutas (JSON con NumpyEncoder)
        with open(os.path.join(directorio_salida, "pipeline_metadata.json"), 'w', encoding='utf-8') as f:
            # 🚀 FIX: Usamos cls=NumpyEncoder para que no explote con los int8 o booleanos de numpy
            json.dump(rutas, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print("  🗺️ Metadata de Rutas exportada a JSON de forma segura.")

    except Exception as e:
        raise RuntimeError(f"🛑 Error crítico en I/O: {e}")

    print("-" * 80)
    print(f"  🛡️ ESTATUS: Activos 100% blindados (Datos + Modelos Previos). Listo para Producción y Optuna.")
    print(f"\n⏱️ I/O completado en {time.time() - inicio_timer:.3f}s")

# ==========================================
# Celda de Ejecución Maestra en tu .ipynb (VÍA MANAGER)
# ==========================================
try:
    if 'manager' not in globals() and 'manager' not in locals():
        raise EnvironmentError("El PipelineManager no está inicializado. Ejecuta las fases previas.")

    if manager.X_train is None or manager.y_train is None:
        raise ValueError("El Manager no tiene cargadas las matrices de entrenamiento.")

    exportar_activos_mlops_integral(
        X_train=manager.X_train, 
        y_train=manager.y_train, 
        X_test=manager.X_test, 
        y_test=manager.y_test, 
        rutas=manager.rutas,
        modelos_preprocesamiento=manager.modelos_preprocesamiento,
        pesos_equidad=manager.pesos_train,
        grupos_validacion=manager.grupos_cv,
        directorio_salida="mlops_activos"
    )

except Exception as e:
    print(f"🛑 Error en la Fase de Exportación: {e}")