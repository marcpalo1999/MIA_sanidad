# Descripción Clínica del Dataset Heart Disease

## Contexto del Estudio

Este dataset proviene del UCI Machine Learning Repository y contiene información de 303 pacientes que fueron evaluados para enfermedad coronaria mediante un protocolo diagnóstico completo que culminó en angiografía coronaria (cateterismo cardíaco).

**Importante**: Estos pacientes ya completaron TODAS las pruebas diagnósticas, incluyendo el cateterismo. El objetivo del modelo es determinar si hubiese sido posible predecir el resultado final usando solo las pruebas menos invasivas realizadas previamente.

---

## Variables del Dataset según Procedimiento Diagnóstico

#### 1. Información Clínica Basal (disponible en consulta inicial)

* **age**: Edad del paciente en años
* **sex**: Sexo (0 = Mujer, 1 = Hombre)
* **cp**: Tipo de dolor torácico reportado por el paciente
  * 1 = Angina típica (dolor opresivo retroesternal con irradiación clásica)
  * 2 = Angina atípica (características atípicas)
  * 3 = Dolor torácico no anginoso
  * 4 = Asintomático
* **trestbps**: Presión arterial sistólica en reposo (mmHg) - tomada al ingreso
* **chol**: Colesterol sérico total (mg/dL) - analítica básica
* **fbs**: Glucemia en ayunas > 120 mg/dL (0 = No, 1 = Sí) - analítica básica

#### 2. ECG en Reposo (prueba no invasiva, bajo coste)

* **restecg**: Hallazgos electrocardiográficos en reposo
  * 0 = Normal
  * 1 = Anormalidad onda ST-T (inversiones de onda T y/o elevación o depresión de ST)
  * 2 = Hipertrofia ventricular izquierda probable o definitiva (criterios de Estes)

#### 3. Prueba de Esfuerzo con ECG (prueba no invasiva, coste moderado)

Primera prueba funcional cuando hay sospecha clínica de enfermedad coronaria.

* **thalach**: Frecuencia cardíaca máxima alcanzada durante la prueba
* **exang**: Angina inducida por ejercicio (0 = No, 1 = Sí)
* **oldpeak**: Magnitud de la depresión del segmento ST inducida por ejercicio (en mm)
* **slope**: Morfología de la pendiente del segmento ST durante ejercicio máximo
  * 1 = Ascendente (generalmente normal)
  * 2 = Plano (sugestivo de isquemia)
  * 3 = Descendente (altamente sugestivo de isquemia)

#### 4. Gammagrafía de Perfusión Miocárdica con Talio-201 (prueba especializada, coste alto, mínimamente invasiva)

Prueba de medicina nuclear que requiere inyección intravenosa de trazador radioactivo. Se realiza cuando la prueba de esfuerzo es equívoca o no concluyente. NO es invasiva en el sentido del cateterismo (no catéteres arteriales, no fluoroscopia prolongada, riesgos mínimos).

* **thal**: Resultado de la gammagrafía de perfusión
  * N = Normal (perfusión miocárdica normal en reposo y estrés)
  * FD = Fixed Defect (defecto fijo - indica infarto previo o cicatriz)
  * RD = Reversible Defect (defecto reversible - indica isquemia inducible, alta probabilidad de enfermedad coronaria significativa)

#### 5. Angiografía Coronaria - Cateterismo Cardíaco (gold standard - objetivo a predecir)

**CRÍTICO**: Esta es la prueba INVASIVA que queremos EVITAR mediante predicción. Requiere cateterización arterial (femoral o radial), fluoroscopia, contraste yodado. Riesgos: complicaciones vasculares, hematomas, nefropatía por contraste, disección arterial, infarto, ictus. Las variables obtenidas aquí NO deben usarse como predictores.

* **ca**: Número de vasos principales con estenosis significativa (0-3) visualizados durante el cateterismo
  * **ATENCIÓN**: Esta variable es DATA LEAKAGE si se usa para predecir `num`
  * Proviene del MISMO procedimiento que queremos evitar
  * Solo se conoce DESPUÉS de realizar el cateterismo
  * **NO debe incluirse en ningún modelo predictivo con utilidad clínica**

* **num**: Diagnóstico definitivo de enfermedad coronaria - **VARIABLE OBJETIVO**
  * **0 = Saludable**: Sin estenosis significativa (<50% de obstrucción luminal)
  * **1 = Enfermedad leve**: Estenosis 50-70% en un vaso
  * **2 = Enfermedad moderada**: Estenosis >70% en un vaso o estenosis 50-70% en dos vasos
  * **3 = Enfermedad severa**: Estenosis significativa en dos vasos principales
  * **4 = Enfermedad muy severa**: Enfermedad de tres vasos o compromiso de tronco de coronaria izquierda

**¿Por qué ca es data leakage?**

`ca` y `num` son dos formas diferentes de codificar el MISMO hallazgo del cateterismo:

1. `ca` = cuántos vasos están obstruidos (información simplificada)
2. `num` = severidad de la enfermedad basada en patrón de obstrucción (información detallada)

Ambos se obtienen al mismo tiempo mirando las mismas imágenes de fluoroscopia. Usar `ca` para predecir `num` es como usar "número de vasos obstruidos" para predecir "grado de enfermedad coronaria" - la primera variable ya contiene esencialmente la respuesta.

---

## Utilidad Clínica del Modelo Predictivo

#### El Problema Clínico Actual

En la práctica clínica habitual, el algoritmo diagnóstico típico es:

**Secuencia de decisión clínica:**

1. **Evaluación inicial**: Clínica + ECG reposo + factores de riesgo
2. **Prueba de esfuerzo**: Si hay sospecha clínica
   * **Negativa + baja probabilidad clínica** → seguimiento conservador
   * **Positiva clara + alta probabilidad clínica** → cateterismo directo
   * **Equívoca o no concluyente** → gammagrafía para aclarar
3. **Gammagrafía**: Solo si prueba esfuerzo no es concluyente
   * **Positiva** → cateterismo
   * **Negativa** → seguimiento
4. **Scores clínicos**: Duke Treadmill Score, Framingham Risk Score como apoyo

**Nota**: La gammagrafía NO se hace rutinariamente. Muchos pacientes van directo de prueba de esfuerzo positiva a cateterismo, especialmente si la probabilidad clínica es alta.

**Limitaciones del enfoque actual:**

* Sensibilidad limitada de prueba de esfuerzo (60-70%) → muchos falsos negativos
* Especificidad limitada (70-80%) → muchos cateterismos "normales" innecesarios
* Criterios de decisión basados en umbrales rígidos que no integran toda la información disponible
* Variabilidad en la interpretación entre médicos
* Decisión de ir directo a cateterismo vs hacer gammagrafía es subjetiva

---

## Estrategias de Modelado Válidas

#### 1 - Modelo Básico (solo variables no invasivas de consulta y esfuerzo)

**Variables predictoras permitidas:**
* age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope

**Variable objetivo:**
* num (simplificado a binario: 0 = sin enfermedad, 1-4 = con enfermedad)

**Utilidad clínica:**

1. Screening inicial para identificar pacientes de muy bajo riesgo que NO necesitan cateterismo
2. Alternativa a scores clínicos tradicionales (Duke Score)
3. Disponible en cualquier consulta de cardiología con ergometría básica
4. Ayudar a decidir: ¿cateterismo directo vs gammagrafía primero vs seguimiento?

**Comparación con práctica actual:**

* **Actual**: Prueba de esfuerzo positiva + criterio clínico → cateterismo directo o gammagrafía (decisión subjetiva)
* **Con modelo**: Prueba de esfuerzo positiva + modelo estratifica riesgo → decisión más objetiva sobre siguiente paso

#### 2 - Modelo Completo Pre-Cateterismo (incluye gammagrafía)

**Variables predictoras permitidas:**
* Todas las anteriores + thal (resultado de gammagrafía)

**Variable objetivo:**
* num (binario)

**Utilidad clínica:**

1. Optimizar indicación de cateterismo en pacientes que YA tienen gammagrafía
2. Integración óptima de múltiples modalidades diagnósticas
3. Útil especialmente en casos "borderline" donde la gammagrafía no es completamente concluyente

**Comparación con práctica actual:**

* **Actual**: Interpretación subjetiva de gammagrafía + criterio clínico → decisión sobre cateterismo
* **Con modelo**: Integración cuantitativa de clínica + esfuerzo + gammagrafía → decisión más precisa

**Nota**: Este modelo tiene menos aplicabilidad que el Modelo Básico porque solo es útil en el subgrupo de pacientes a los que se les hizo gammagrafía (no todos).

#### 3 - Incluir ca como predictor (INCORRECTA)

**Lo que NO debemos hacer:**

* Incluir ca como predictor produciría un modelo con accuracy artificialmente muy alto (probablemente >95%)
* NO tiene ninguna utilidad clínica (si ya sabes ca, ya hiciste el cateterismo)
* Es metodológicamente incorrecto y engañoso
* **Solo sirve para análisis exploratorio** (ej: verificar que ca está fuertemente asociado con num, validación de consistencia de datos)

---

### Impacto Clínico Esperado

#### Escenario actual sin modelo ML:

* 100 pacientes con prueba de esfuerzo positiva
* **Decisión actual**: 
  * ~70 pacientes → cateterismo directo (criterio subjetivo: "parece alta probabilidad")
  * ~30 pacientes → gammagrafía primero (criterio subjetivo: "no estoy seguro")
* **Resultado**: De los 70 que fueron directo a cateterismo, solo 35-40 tienen enfermedad significativa
* **Problema**: 30-35 cateterismos innecesarios
* **Coste**: ~70 × 3,000€ = 210,000€ solo en cateterismos
* **Riesgos**: 70 pacientes expuestos a complicaciones

#### Escenario con Modelo Básico (Estrategia 1):

* 100 pacientes con prueba de esfuerzo positiva
* **Modelo estratifica objetivamente**:
  * Probabilidad <10%: 20 pacientes → seguimiento conservador (evitar cateterismo)
  * Probabilidad 10-40%: 30 pacientes → gammagrafía primero
  * Probabilidad 40-70%: 35 pacientes → considerar cateterismo o gammagrafía según contexto clínico
  * Probabilidad >70%: 15 pacientes → cateterismo directo (muy alta probabilidad)
* **Resultado esperado**:
  * Reducción de 25-30 cateterismos directos innecesarios
  * Aumento selectivo de uso de gammagrafía (más eficiente)
  * Sensibilidad mantenida >90% (no perder casos reales)
  * Ahorro: ~25 × 3,000€ = 75,000€ por cada 100 pacientes
  * Reducción significativa de exposición a riesgos

#### Escenario con Modelo Completo (Estrategia 2):

* 30 pacientes que ya tienen gammagrafía (del escenario anterior)
* **Modelo integra**: clínica + esfuerzo + gammagrafía
* **Resultado esperado**:
  * De los 30 con gammagrafía, el modelo identifica mejor quién REALMENTE necesita cateterismo
  * Reducción adicional de 5-8 cateterismos innecesarios en este subgrupo
  * Especialmente útil cuando gammagrafía muestra defecto leve/moderado (zona gris)

**Impacto total combinando ambas estrategias**: Reducción de ~30-35% de cateterismos innecesarios, con ahorro de ~100,000€ por cada 100 pacientes con prueba esfuerzo positiva.

---

## Simplificación de la Variable Objetivo

Para facilitar el modelado, la variable num frecuentemente se convierte a clasificación binaria:

* **0 = "No enfermedad"**: Paciente saludable (num = 0)
* **1 = "Enfermedad presente"**: Paciente con enfermedad coronaria significativa (num = 1, 2, 3 o 4)

**Justificación clínica**: 

1. La decisión crítica es "¿necesita cateterismo?" (sí/no), no "¿qué grado exacto de severidad tiene?"
2. El grado exacto de severidad se determina definitivamente durante el cateterismo
3. Simplifica el modelado y mejora la interpretabilidad clínica

---

## Estructura de los Datos

#### Variables utilizables para modelado:

**Modelo básico (11 variables predictoras):**
* age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope

**Modelo completo (12 variables predictoras):**
* Las 11 anteriores + thal

**Variables que NO deben usarse como predictores:**
* ca (data leakage - proviene del mismo procedimiento que queremos predecir)

**Variable objetivo:**
* num (enfermedad coronaria según angiografía)