"""Inserta en el notebook ya ejecutado las celdas de interpretación, escritas a partir de las cifras
reales de la corrida 2026-09-25T23-53. Las celdas Markdown no requieren reejecutar el notebook.

Uso:  python ml/interpretar.py
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"

TEXTOS = {
"tabla_revision": None,  # se genera desde artefactos/ en tiempo de ejecución
"eda": """
**Longitud.** Las consultas son **cortas y muy asimétricas**: en entrenamiento la mediana es de
**10 palabras (47 caracteres)** y la media de 11.95 palabras, pero la cola llega a **79 palabras
(433 caracteres)**. La media supera a la mediana en ambas métricas y la desviación estándar
(7.89 palabras) es casi el 66 % de la media: una minoría de consultas largas, con contexto
narrativo («I just made one and it doesn't seem to be working…»), convive con muchas de 5–10
palabras. El conjunto de prueba tiene la misma forma con consultas algo más cortas (mediana 9
palabras), así que no hay un desplazamiento de distribución preocupante. La longitud depende de la
intención: `card_acceptance` promedia 7.5 palabras y `transfer_not_received_by_recipient` 17.5 —
las intenciones sobre transferencias necesitan explicar una historia.

**Vocabulario.** Tras la limpieza, las palabras dominantes son **card (3 591), account (1 716),
money, transfer y top**: el dominio gira alrededor de cinco objetos (tarjeta, cuenta, dinero,
transferencia, recarga). Los bigramas y trigramas lo confirman y ya anticipan las confusiones:
*exchange rate*, *new card*, *virtual card*, *card payment*, *disposable virtual card*,
*get money back*. Muchas intenciones distintas **comparten exactamente estas expresiones**; por
ejemplo, *virtual card* aparece en `getting_virtual_card`, `get_disposable_virtual_card`,
`virtual_card_not_working` y `disposable_card_limits`. Un modelo de bolsa de palabras lo tendría
difícil; se necesita un modelo que lea la frase completa (*how many* → límites; *not working* →
fallo).

**Limpieza.** Eliminar stopwords de NLTK borra palabras con carga semántica para este problema:
*not*, *why*, *how*, *can't*, *still* (por ejemplo, «How old do you have to be?» se reduce a
«old»). Por eso la versión limpia se usó solo para el análisis y el Transformer recibe el texto
original.

**Balance.** El conjunto de entrenamiento está **moderadamente desbalanceado**: de **35**
consultas (`contactless_not_working`) a **187** (`card_payment_fee_charged`), una razón de
**5.3×** y un coeficiente de variación de 0.25. Aun así la entropía normalizada es **0.992**
(1 = uniforme): ninguna clase domina y ninguna es marginal. El conjunto de **prueba está
perfectamente balanceado (40 consultas por clase)**, así que accuracy y F1 macro miden casi lo
mismo y no hace falta reponderar clases; basta con vigilar el recall de las clases pequeñas.

**Implicaciones para el modelo de clasificación** (lo que este análisis decide antes de entrenar):

1. **Consultas cortas** (mediana de 13 tokens, máximo 96) → `max_length = 128` no trunca ninguna consulta
   y un lote de 32 cabe holgado en memoria.
2. **Vocabulario compartido entre intenciones** (*virtual card*, *exchange rate*, *top up*) → hace falta un
   modelo contextual, no una bolsa de palabras. Y se anticipa que las familias transferencias, recargas y
   tarjetas concentrarán las confusiones: se comprueba en la sección 3.6.
3. **Las stopwords llevan intención** (*not*, *why*, *still*) → el Transformer recibe el texto original; la
   limpieza es solo para este análisis.
4. **Desbalance moderado en train (5.3×) y prueba balanceada** → la época se elige por **F1 macro**, que
   pesa igual las 77 clases, y en la sección 3.6 se mide si el tamaño de la clase explica los errores.
""",
"causas": """
**¿Lo anticipaba el EDA? Dos hipótesis medidas en la celda anterior.**

- **H1 — el desbalance causa los errores: se rechaza.** La correlación entre el número de ejemplos de
  entrenamiento de una clase y su F1 es **ρ = −0.14 (p = 0.21)**: no hay relación significativa. Las siete
  clases con más errores tienen de media **143 ejemplos**, *más* que el promedio (130), y las siete más
  pequeñas —como `contactless_not_working`, con solo 35— logran un F1 entre **0.886 y 1.0**.
- **H2 — el solapamiento de n-gramas causa los errores: se confirma.** Los pares de clases que se confunden
  comparten **9 veces más bigramas frecuentes** que los que nunca se confunden (Jaccard 0.042 frente a
  0.005; ρ = 0.31, p ≈ 10⁻⁶⁴). `verify_my_identity` ↔ `why_verify_identity` comparten *verify identity*,
  *identity check* y *need verify*; `card_arrival` ↔ `card_delivery_estimate`, *new card* y *card delivered*.
  **Matiz:** el par más confundido, `top_up_failed` ↔ `top_up_reverted` (10 errores), no comparte bigramas
  frecuentes: ahí la confusión es semántica («fallida» y «revertida» se cuentan con palabras distintas pero
  describen lo mismo para el cliente), algo que un n-grama no captura.

**Las clases problemáticas fallan hacia sus vecinas semánticas, no al azar.** Los siete destinos
principales tienen una similitud léxica TF-IDF con la clase real que está en el **percentil
94.8–99.9** de los 2 926 pares posibles; en seis de los siete casos, por encima del percentil 99.
En todo el conjunto, la correlación de Spearman entre similitud léxica de un par y número de
errores entre ellos es **ρ = 0.28 (p ≈ 10⁻⁵⁴)**: cuanto más vocabulario comparten dos intenciones,
más se confunden. El 57.7 % de los 208 errores ocurre, además, **dentro de la misma familia**
(transferencias, recargas, tarjetas, cambio de divisa, identidad).

Tres causas concretas, con ejemplos del propio conjunto de prueba:

1. **Intenciones que difieren en un matiz temporal o de estado.** `pending_transfer` (13 errores,
   la peor) se va a `balance_not_updated_after_bank_transfer` y a
   `transfer_not_received_by_recipient`; «When will my transfer be available in my account.» o
   «I transferred some money but it is yet to arrive» describen la misma situación vista desde
   ángulos distintos (pendiente / saldo sin actualizar / no llegó). Lo mismo con `top_up_failed`
   ↔ `top_up_reverted` (6 errores en un sentido y 4 en el otro): «fallida» y «revertida» se
   expresan con las mismas palabras.
2. **Pares casi sinónimos.** `why_verify_identity` ↔ `verify_my_identity` tiene la similitud más
   alta de todas las mostradas (0.30, percentil 99.9): «What other methods are there to verify my
   identity?» contiene *verify my identity* literalmente.
3. **Etiquetas dudosas o consultas genuinamente ambiguas.** «How do I top up my card?» está
   etiquetada como `transfer_into_account` y «How long can an EU transfer take?» como
   `pending_transfer`, cuando `topping_up_by_card` y `transfer_timing` (lo que predijo el modelo)
   son al menos igual de correctas. Parte del «error» es ruido de anotación, un problema conocido
   de BANKING77 y del que ningún ajuste de hiperparámetros puede librarse.

`declined_transfer` es un caso distinto: **precision 1.0 y recall 0.75**. El modelo nunca la
predice por error, pero la infrautiliza: 10 de sus consultas («I can't transfer money from my
account») se reparten entre `declined_card_payment` y otras de fallo, porque *declined* y
*can't* aparecen en varias intenciones de rechazo.
""",
"transformer": """
- **El transfer learning funciona muy bien con poco dato.** Con solo ~117 ejemplos por clase y
  **8.4 minutos** de entrenamiento en un portátil (Apple M5 Pro, MPS), DistilRoBERTa alcanza
  **93.25 % de accuracy y 0.932 de F1 macro** en las 3 080 consultas de prueba, en línea con lo
  publicado para BANKING77 (≈ 93 % con BERT-base, Casanueva et al., 2020) y con la mitad de capas.
- **El entrenamiento es sano.** La pérdida de validación baja de 1.26 a 0.29 en la **época 6**
  (la elegida por *early stopping* sobre F1 macro: 0.930) y después sube ligeramente mientras la
  de entrenamiento sigue cayendo (0.027): a partir de ahí el modelo empieza a memorizar. Quedarse
  con la mejor época y no con la última evita ese sobreajuste; la métrica en prueba (0.932)
  coincide con la de validación, lo que indica que la validación era representativa.
- **El desempeño es muy desigual entre clases.** Nueve clases tienen F1 = 1.0 (las siete de la
  tabla son las primeras en el orden oficial del dataset entre esas nueve empatadas) y cuatro
  quedan por debajo de 0.85, con el mínimo en `balance_not_updated_after_bank_transfer` (0.76).
  Las siete peores concentran el **28 % de los errores** siendo el 9 % de las clases. Las mejores
  tienen vocabulario exclusivo (*age*, *Apple Pay*, *ATM*, *terminate*, *passcode*); las peores
  comparten vocabulario con sus vecinas.
- **La confianza del modelo es informativa, pero no fiable del todo.** La confianza media es
  0.97 en los aciertos y 0.75 en los errores, pero **el 35 % de los errores se comete con
  confianza > 0.90**. En producción convendría un umbral para derivar a una persona las consultas
  dudosas, sabiendo que no atrapará todos los errores.
- **Qué mejoraría el resultado.** Más que más épocas: (1) revisar y corregir las etiquetas
  dudosas de las familias transferencias/recargas; (2) aumentar datos en los pares confundidos con
  paráfrasis que marquen el matiz (pendiente vs. no recibido); (3) un modelo más grande
  (RoBERTa-base) o un ensamble; (4) calibrar las probabilidades (p. ej. *temperature scaling*).
""",
"mejoras": """
Cada mejora sale de un hallazgo medido en este notebook, no de una lista genérica:

| Hallazgo (sección) | Ajuste propuesto | Por qué | Referencia |
|---|---|---|---|
| El desbalance de train (5.3×) **no** explica los errores: ρ = −0.14 (3.6, H1) | **No** priorizar *class weights*; si se prueban, pérdida ponderada por el número efectivo de muestras | Reponderar clases pequeñas que ya aciertan (F1 ≥ 0.886) movería la frontera a costa de las grandes sin atacar la causa | Cui et al. (2019) |
| Los pares confusos comparten 9× más bigramas (3.6, H2) | ***Data augmentation* textual dirigida** a esos pares: paráfrasis que marquen el matiz (pendiente / no recibido / saldo sin actualizar) con *back-translation* y operaciones EDA | Aumentar datos al azar no ayuda; ampliar la frontera entre clases vecinas sí | Wei y Zou (2019); Sennrich et al. (2016) |
| ≈ 7 de los 20 errores revisados tienen etiqueta dudosa (4.6) | **Detectar etiquetas ruidosas** con *confident learning* y corregirlas antes de reentrenar | Parte del techo del modelo es ruido de anotación, no capacidad | Northcutt et al. (2021) |
| El 35 % de los errores se comete con confianza > 0.90 (3.7) | **Calibrar las probabilidades** con *temperature scaling* y derivar a una persona las consultas bajo un umbral | Sin calibrar, la confianza no sirve para abstenerse | Guo et al. (2017) |
| Las stopwords llevan la intención (2.2) | Mantener el **texto original** como entrada del Transformer (ya aplicado) | RoBERTa se preentrenó con texto sin limpiar; quitar *not* o *why* cambia la intención | Liu et al. (2019) |
| Confusión semántica residual sin solapamiento léxico: `top_up_failed` ↔ `top_up_reverted` (3.6) | Modelo de más capacidad (RoBERTa-base) o **ensamble** | Más capas de atención separan mejor matices que no dependen de palabras compartidas | Liu et al. (2019); Sanh et al. (2019) |

**El vínculo EDA → ajustes, en una línea:** el análisis exploratorio señaló dos sospechosos —el desbalance y el
vocabulario compartido—; la evaluación descartó el primero y confirmó el segundo, así que la inversión va a
los datos de los pares confusos y a la calidad de las etiquetas, no a reponderar clases.
""",
"calibracion": """
**Resultado de la calibración (5 consultas × 3 configuraciones):**

- **C (temperatura 1.0, 150 tokens)** es la peor: solo **2 de 5** respuestas se quedan en 1–2
  oraciones, es la más larga (64 tokens de media) y es la que más inventa en contenido: dos de sus
  cinco respuestas atribuyen el error a que **el cliente «escribió mal»** una palabra
  («probably mistyped 'exchange rate' as 'exchange charge'»), algo que no aparece en ninguna
  consulta. Es la alucinación típica de una temperatura alta: el modelo elige continuaciones poco
  probables y construye una historia.
- **A (codiciosa) y B (temperatura 0.3)** respetan el formato en **5 de 5**. B genera algo menos
  de vocabulario ajeno a la consulta (0.65 frente a 0.69) y ligeramente más breve (1.4 oraciones
  de media), así que la regla fijada de antemano elige **B**: `temperature=0.3`, `top_p=0.9`,
  `max_new_tokens=60`, `repetition_penalty=1.15`.

Dos lecciones de la calibración: (1) **la temperatura controla la invención más que la longitud**:
C no solo es más larga, cambia el tipo de explicación; (2) **`max_new_tokens` es un límite duro,
no un objetivo**: con 60 tokens, **4 de las 20 respuestas finales quedaron cortadas a media
frase** (se ven como la #2, #3 o #9). El recorte a dos oraciones no puede reparar una oración
incompleta; en una iteración siguiente habría que subir a ~80 tokens y pedir explícitamente
«one sentence».

La métrica automática de «palabras ajenas» resultó **poco discriminante** (0.61–0.69 en las tres):
el LLM usa muchas palabras del nombre de las clases y de su propio vocabulario explicativo. Por eso
la validación decisiva es la **revisión manual** de la sección siguiente.
""",
"razones": """
**Veredicto de la revisión manual: 3 pertinentes, 9 parciales y 8 alucinadas de 20** (15 %,
45 % y 40 %). Las razones que da el LLM se agrupan así:

| Razón que da el LLM | Nº | Qué dice | ¿Es cierta? |
|---|---|---|---|
| **Solapamiento léxico** | 8 | «la palabra X de la consulta apunta a la clase predicha» | 3 veces sí (#8 *transfer*, #11 *top up my card*, #15 *declined*); 4 veces **cita palabras que no están** en la consulta (#1 *card*, #2 *transaction*, #3 y #9 afirman que *exchange rate* no aparece cuando sí aparece) |
| **Genérica** | 7 | «la clase predicha es más específica / directamente relacionada» | No es falsa, pero no explica nada; es la misma frase comodín en 7 consultas distintas |
| **Frecuencia supuesta** | 3 | «la clase predicha es más común» (#14, #18, #20) | Inventada: el modelo no tiene ningún dato de frecuencias de clase |
| **Ambigüedad real** | 2 | «la clase correcta es más general / se puede interpretar de otra forma» (#5, #6) | Plausible en #5; en #6 intercambia las dos etiquetas |

Tres patrones de alucinación que se repiten:

1. **Citar evidencia inexistente** — el fallo más grave, porque precisamente se le pidió referirse
   solo a palabras de la consulta y el texto «suena» verificable.
2. **Intercambiar la clase real y la predicha** (#6, #20): con dos etiquetas parecidas en el
   prompt, el modelo pierde cuál es cuál.
3. **Inventar entidades** — una intención «view PIN» (#17) o «currency conversion» (#6) que no
   existen en el esquema.

Hallazgo más interesante: **en al menos 7 de los 20 casos el clasificador probablemente tenía
razón** y la etiqueta del dataset es discutible (#3, #4, #9, #11, #18, #19 y, parcialmente, #15:
«whats your exchange rate» etiquetada como `exchange_charge`, «My card is being declined» como
`reverted_card_payment?`). El LLM **no detectó ni uno**: el prompt afirma que la predicción «is
WRONG» y el modelo acepta la premisa y la racionaliza.
""",
"llm": """
- **Un LLM de 7 B produce explicaciones fluidas, pero no fieles.** Solo el **15 %** de las
  explicaciones fue plenamente pertinente y el **40 %** afirmó algo falso. La fluidez es
  precisamente el riesgo: todas «suenan» razonables y habría sido fácil aceptarlas sin revisarlas.
  Sin la verificación manual, este análisis habría concluido lo contrario.
- **La calibración sí importa, y funciona en la dirección esperada.** Bajar la temperatura de 1.0 a
  0.3 eliminó las historias inventadas sobre el cliente («escribió mal») y llevó el cumplimiento
  del formato de 2/5 a 5/5; el límite de tokens acotó la longitud (20/20 en ≤ 2 oraciones tras el
  postproceso), a costa de 4 respuestas cortadas. Pero **ninguna combinación de parámetros hace
  que el modelo sepa algo que no sabe**: Falcon no ve los pesos del clasificador ni sus
  atenciones, así que su «explicación» es una conjetura *post hoc* a partir del texto, no una
  explicación causal del modelo.
- **El encuadre del prompt condiciona la respuesta.** Decirle al LLM que la predicción es
  incorrecta le impidió detectar el hallazgo más valioso: que varias «equivocaciones» son en
  realidad etiquetas dudosas. Una versión mejor del prompt debería (a) no presuponer quién tiene
  razón y pedir primero «¿qué etiqueta encaja mejor y por qué?», (b) dar las **definiciones** de
  las dos intenciones, (c) incluir 2–3 ejemplos resueltos (*few-shot*) y (d) exigir que cite entre
  comillas las palabras de la consulta, lo que permite **verificar automáticamente** que existen.
- **Uso recomendado.** Como herramienta de **triaje** para un analista humano — sugerir hipótesis
  sobre grupos de errores, detectar pares de clases confusas, redactar un primer borrador — sí es
  útil y barato (≈ 4 s por explicación en un portátil). Como explicación final para un cliente o
  un auditor, **no**, a menos que se combine con métodos de atribución que sí miran dentro del
  clasificador (SHAP, gradientes integrados, pesos de atención) y con verificación automática de
  las citas.
""",
"generales": """
1. **El Transformer resuelve bien la tarea.** DistilRoBERTa afinado alcanza **93.25 % de accuracy
   y 0.932 de F1 macro** sobre 77 intenciones con unos 117 ejemplos por clase y 8.4 minutos de
   entrenamiento, al nivel de lo publicado con modelos del doble de tamaño. El análisis
   exploratorio explicaba de antemano por qué un modelo contextual era necesario: consultas cortas
   que comparten vocabulario (*card*, *transfer*, *top up*) y cuyo sentido lo deciden palabras que
   un preprocesado clásico eliminaría (*not*, *why*, *still*).
2. **Los errores tienen estructura, y el EDA la anticipaba a medias.** No se reparten al azar: se
   concentran en familias de intenciones con vocabulario compartido (ρ = 0.28 entre similitud TF-IDF y
   confusión; 9× más bigramas compartidos en los pares confusos; 57.7 % de los errores dentro de la misma
   familia). En cambio, el desbalance que también mostraba el EDA **no** los explica (ρ = −0.14, p = 0.21).
   El techo práctico del modelo lo pone tanto la **calidad de las etiquetas** como la arquitectura.
3. **El LLM ayuda a pensar, no a explicar.** Con un prompt estructurado y temperatura baja,
   Falcon-7b-instruct produce explicaciones breves y en formato, pero solo 3 de 20 fueron
   plenamente fieles y 8 contenían afirmaciones falsas. La calibración reduce la invención
   (temperatura) y controla la longitud (`max_new_tokens`), pero no convierte una conjetura en una
   explicación verificada. **La validación manual no es opcional.**
4. **Lección metodológica.** El resultado más valioso del ejercicio no lo dio ninguno de los dos
   modelos, sino la revisión humana de sus salidas: descubrir que parte de los «errores» del
   clasificador son aciertos frente a etiquetas mal puestas.

**Ejemplos del experimento que lo sostienen:**

- *EDA → Transformer*: `verify_my_identity` y `why_verify_identity` comparten los bigramas *verify
  identity* e *identity check*, y el modelo las confundió 6 veces entre sí.
- *Transformer → calidad de las etiquetas*: «How do I top up my card?» está etiquetada como
  `transfer_into_account`; el modelo predijo `topping_up_by_card` con 98.9 % de confianza, y es lo más
  razonable.
- *LLM*: la explicación n.º 1 afirma que la consulta contiene la palabra *card* y no la contiene
  (alucinación); la n.º 11 sí señala correctamente *top up my card* como pista.
- *Limpieza*: «How old do you have to be?» queda reducida a «old» al quitar stopwords; por eso el modelo
  recibe el texto original.

**Limitaciones.** Una sola corrida con una semilla (sin intervalos de confianza); la validación
manual la hizo una sola persona; las 20 muestras son las de mayor confianza, no una muestra
aleatoria, así que sobrerrepresentan los casos difíciles; Falcon se ejecutó en `float16` en MPS y
no cuantizado, como sería en Colab, lo que puede variar ligeramente sus salidas.

Todos los resultados de esta ejecución se publicaron en <https://distilroberta.iagentek.com.mx>,
donde además se puede clasificar una consulta nueva en vivo.
""",
}

TABLA_PROMPT = """

**El prompt, en inglés (tal como se envía) y en español.** Se escribe en inglés porque las consultas y las
etiquetas del dataset están en inglés y Falcon-7b-instruct se ajustó sobre todo con instrucciones en inglés;
mezclar idiomas en el prompt empeora la adherencia al formato.

| Inglés (enviado a Falcon) | Español |
|---|---|
| You are a banking customer-support analyst who audits an automatic intent classifier. | Eres un analista de atención a clientes bancarios que audita un clasificador automático de intenciones. |
| The classifier read the customer query below and predicted an intent that is WRONG. | El clasificador leyó la consulta del cliente de abajo y predijo una intención que es INCORRECTA. |
| In at most two short sentences, explain why the classifier probably chose the predicted intent instead of the correct one. | En máximo dos oraciones cortas, explica por qué el clasificador probablemente eligió la intención predicha en lugar de la correcta. |
| Refer only to words that appear in the query. Do not invent facts, do not give advice to the customer. | Refiérete solo a palabras que aparecen en la consulta. No inventes hechos y no des consejos al cliente. |
| Customer query: "{texto}" | Consulta del cliente: «{texto}» |
| Predicted intent: {pred} | Intención predicha: {pred} |
| Correct intent: {real} | Intención correcta: {real} |
| Explanation: | Explicación: |
"""

REFERENCIAS_NUEVAS = [
    ("- Devlin, J., Chang,", "- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective\n  Number of Samples. *CVPR*.\n"),
    ("- Holtzman, A.,", "- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.\n  *ICML*.\n"),
    ("- Sanh, V.,", "- Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident Learning: Estimating Uncertainty in Dataset\n  Labels. *Journal of Artificial Intelligence Research*, 70, 1373–1411.\n"),
    ("- Vaswani, A.,", "- Sennrich, R., Haddow, B., & Birch, A. (2016). Improving Neural Machine Translation Models with\n  Monolingual Data. *ACL*. (Back-translation.)\n"),
    ("- Wolf, T.,", "- Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text\n  Classification Tasks. *EMNLP-IJCNLP*.\n"),
]

CORRECCIONES = {
    "| Tamaño de lote | 32 | cabe holgado en memoria con `max_length` 64 |":
        "| Tamaño de lote | 32 | cabe holgado en memoria: las consultas son cortas (mediana de 13 tokens) |",
    "  dos intenciones), `etiqueta_dudosa` (la etiqueta del dataset es discutible), `generica`\n  (no da una razón concreta) y `otra`.":
        "  dos intenciones), `etiqueta_dudosa` (la etiqueta del dataset es discutible), `generica`\n  (no da una razón concreta), `frecuencia_supuesta` (atribuye el error a que una clase es «más\n  común», dato que el LLM no tiene) y `otra`.",
}

import json

def tabla_revision():
    art = NB.parents[1] / "artefactos"
    exp = {e["orden"]: e for e in json.load(open(art / "explicaciones_crudas.json"))}
    filas = ["**Tabla completa de la revisión manual**", "",
             "| # | Consulta | Real → predicha | Explicación del LLM | Veredicto | Razón | Nota de revisión |",
             "|---|---|---|---|---|---|---|"]
    esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
    for r in json.load(open(art / "revision_manual.json")):
        e = exp[r["orden"]]
        filas.append(f"| {r['orden']} | {esc(e['texto'])} | `{e['real']}` → `{e['predicha']}` | {esc(e['explicacion'])} | "
                     f"**{r['veredicto']}** | {r['razon_categoria']} | {esc(r['nota_revision'])} |")
    return "\n".join(filas)

TEXTOS["tabla_revision"] = tabla_revision()

nb = nbf.read(NB, as_version=4)
# 4.2: tabla bilingüe del prompt (una sola vez)
for c in nb.cells:
    if c.cell_type == "markdown" and c.source.startswith("### 4.2 Diseño del prompt") and "en español" not in c.source:
        c.source += TABLA_PROMPT.rstrip("\n")
    if c.cell_type == "markdown" and c.source.startswith("## 6. Referencias"):
        for antes, ref in REFERENCIAS_NUEVAS:
            if ref.split("(")[0] not in c.source:
                c.source = c.source.replace(antes, ref + antes, 1)
# 3.8: mejoras técnicas, justo antes de la sección 4 (una sola vez)
if not any(c.metadata.get("interpretacion") == "mejoras" for c in nb.cells):
    i = next(k for k, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## 4."))
    titulo = nbf.v4.new_markdown_cell("### 3.8 Mejoras técnicas propuestas: del diagnóstico al ajuste")
    cuerpo = nbf.v4.new_markdown_cell("")
    cuerpo.metadata["interpretacion"] = "mejoras"
    nb.cells[i:i] = [titulo, cuerpo]
# Insertar la celda de la tabla tras la celda de revisión (una sola vez)
if not any(c.metadata.get("interpretacion") == "tabla_revision" for c in nb.cells):
    i = next(k for k, c in enumerate(nb.cells) if "pausa_revision" in c.metadata.get("tags", []))
    nueva = nbf.v4.new_markdown_cell("")
    nueva.metadata["interpretacion"] = "tabla_revision"
    nb.cells.insert(i + 1, nueva)
hechas = set()
for celda in nb.cells:
    if celda.cell_type != "markdown":
        continue
    marca = celda.metadata.get("interpretacion")
    if marca in TEXTOS:
        celda.source = TEXTOS[marca].strip("\n")
        hechas.add(marca)
    for viejo, nuevo in CORRECCIONES.items():
        if viejo in celda.source:
            celda.source = celda.source.replace(viejo, nuevo)
            hechas.add(viejo[:30])
nbf.write(nb, NB)
print(f"{len(hechas)} cambios:", sorted(hechas))
