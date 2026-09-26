"""Inserta en el notebook ya ejecutado las celdas de interpretación, escritas a partir de las cifras
reales de la corrida v2 (ejecución limpia). Las celdas Markdown no requieren reejecutar el notebook.

Uso:  python ml/interpretar.py
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks" / "SCA_Actividad2_DistilRoBERTa_Banking77.ipynb"

TEXTOS = {
"fig_longitud": """
**Lectura de la figura.** Las dos distribuciones son asimétricas a la derecha: la mayoría de las consultas tiene
entre 7 y 13 palabras (primer y tercer cuartil de train) y una cola larga llega a 79 palabras. Train y test se
superponen casi por completo, así que la prueba mide al modelo en el mismo tipo de texto con el que aprende.
Las intenciones sobre transferencias son las más largas (hasta 17.5 palabras de media) porque el cliente cuenta
una historia; las de aceptación de tarjeta o PIN, las más cortas.
""",
"fig_ngramas": """
**Lectura de la figura.** Cinco palabras dominan el vocabulario limpio (*card*, *account*, *money*, *transfer*,
*top*) y los bigramas y trigramas más frecuentes son justo las expresiones que comparten varias intenciones
(*exchange rate*, *virtual card*, *new card*, *disposable virtual card*). Es la primera señal de que las
confusiones se darán entre intenciones vecinas, algo que se mide en la sección 3.6.
""",
"fig_nube": """
**Lectura de la figura.** La nube usa exactamente las frecuencias de la tabla anterior (150 palabras), sin fusionar
bigramas: *card* domina con claridad, seguida de *account*, *money*, *transfer* y *top*. Visualmente confirma que el
dataset gira alrededor de pocos objetos —tarjeta, cuenta, dinero, transferencia y recarga— y que las 77 intenciones
se distinguen más por los verbos y los matices (*not working*, *pending*, *declined*) que por los sustantivos.
""",
"fig_balance": """
**Lectura de la figura.** Las barras bajan de forma gradual, sin saltos: 11 clases tienen menos de 100 ejemplos, 42
entre 100 y 149 y 24 al menos 150 (media 130). El desbalance es moderado y continuo, no hay clases marginales.
""",
"eda": """
**Longitud.** Las consultas son **cortas y muy asimétricas**: en entrenamiento la mediana es de **10 palabras
(47 caracteres)** y la media de 11.95, pero la cola llega a **79 palabras (433 caracteres)**. La desviación estándar
(7.89 palabras) es casi el 66 % de la media: una minoría de consultas largas, con contexto narrativo, convive con
muchas de 5 a 10 palabras. La prueba tiene la misma forma, con consultas algo más cortas (mediana de 9 palabras),
así que no hay un desplazamiento de distribución preocupante.

**Vocabulario.** Tras la limpieza dominan **card (3 591), account (1 716), money, transfer y top**. Los bigramas y
trigramas más frecuentes (*exchange rate*, *new card*, *virtual card*, *disposable virtual card*, *get money back*)
son expresiones que **varias intenciones comparten**, así que el sustantivo no basta para decidir: lo decide el
resto de la frase (*how many* → límites; *not working* → fallo).

**Limpieza.** La lista de stopwords de NLTK (198 palabras) incluye palabras que llevan la intención: *not*, *no*,
*why*, *how* y *up*. «My card is not working» queda en «card working» y «How do I top up my card?» en «top card»
(la celda de 2.2 lo muestra). *can't* también desaparece, pero no por ser stopword: la regla de caracteres
especiales lo parte en «can» y «t», que sí lo son. Por eso la versión limpia se usa solo para este análisis.

**Balance.** Entrenamiento **moderadamente desbalanceado**: de **35** consultas (`contactless_not_working`) a
**187** (`card_payment_fee_charged`), una razón de **5.3×** y un coeficiente de variación de 0.25; la entropía
normalizada es **0.992** (1 = uniforme). La **prueba está perfectamente balanceada (40 por clase)**, así que accuracy
y F1 macro medirán casi lo mismo.

**Implicaciones para el modelo de clasificación** (lo que este análisis decide antes de entrenar):

1. **Consultas cortas** → la longitud máxima se fija con la distribución de tokens (sección 3.1) sin truncar
   ninguna consulta, y un lote de 32 cabe holgado en memoria.
2. **Vocabulario compartido entre intenciones** → hace falta un modelo contextual, no una bolsa de palabras; y se
   anticipa que transferencias, recargas y tarjetas concentrarán las confusiones (**hipótesis H2**, sección 3.6).
3. **Las stopwords llevan intención** → el Transformer recibe el texto original.
4. **Desbalance moderado en train y prueba balanceada** → la época se elige por **F1 macro**, que pesa igual las 77
   clases, y se comprueba si el tamaño de la clase explica los errores (**hipótesis H1**, sección 3.6).
""",
"fig_curvas": """
**Lectura de la figura.** La pérdida de validación baja con fuerza hasta la época 3 y alcanza su mínimo en la
**época 6** (0.295); después sube ligeramente mientras la de entrenamiento sigue cayendo (0.027 en la época 8): a
partir de ahí el modelo empieza a memorizar. El *early stopping* conserva la época 6, la de mayor F1 macro en
validación (0.930).
""",
"fig_confusion": """
**Lectura de la figura.** La diagonal concentra casi todo (2 872 aciertos de 3 080). Los 208 errores no se reparten
al azar: el mapa de solo errores muestra celdas aisladas y pequeños grupos junto a la diagonal, que corresponden a
familias de intenciones con nombres y vocabulario parecidos. Los pares más confundidos (tabla) son de
transferencias, recargas, identidad y divisas; ningún par supera los 6 errores en un sentido.
""",
"fig_f1": """
**Lectura de la figura.** El F1 va de 0.762 a 1.000: 29 clases superan 0.95, 30 están entre 0.90 y 0.95 y 18 por debajo
de 0.90. Hay **empates** que el orden resuelve y conviene declarar: 9 clases tienen F1 = 1.000 (se muestran las 7
primeras en el orden oficial) y tres clases empatan con 6 errores en el corte de las 7 peores
(`get_disposable_virtual_card` queda fuera por tener mayor F1).
""",
"causas": """
**¿Lo anticipaba el EDA? Las dos hipótesis, medidas en la celda anterior.**

- **H1 — el desbalance causa los errores: no hay evidencia a favor.** La correlación entre los ejemplos de
  entrenamiento de una clase y su F1 es **ρ = −0.14 (p = 0.21)**, no significativa. Las siete clases con más errores
  tienen de media **143 ejemplos**, *más* que el promedio (130), y las siete más pequeñas —como
  `contactless_not_working`, con 35— logran un F1 entre **0.886 y 1.0**. Un resultado no significativo no prueba que el
  desbalance no influya, pero indica que no es el factor principal.
- **H2 — el solapamiento de n-gramas se asocia a los errores: se confirma.** Los pares que se confunden comparten
  **9 veces más bigramas frecuentes** que los que nunca se confunden (Jaccard 0.042 frente a 0.005; ρ = 0.31,
  p ≈ 10⁻⁶⁴). `verify_my_identity` ↔ `why_verify_identity` comparten *verify identity*, *identity check* y *need
  verify*; `card_arrival` ↔ `card_delivery_estimate`, *new card* y *card delivered*. Es una asociación, no una prueba
  causal, pero coincide con lo que anticipó el EDA.

**Las clases problemáticas fallan hacia sus vecinas, no al azar.** Los destinos principales de las siete peores tienen
una similitud TF-IDF con su clase real en el **percentil 94.8–99.9** de los 2 926 pares, y la correlación general entre
similitud y errores es **ρ = 0.28**. Además, el **68.3 % de los errores (142 de 208) ocurre dentro de la misma familia**
de intenciones (regla por palabra clave del nombre, documentada en la celda).

Tres causas concretas, con errores reales de la salida anterior:

1. **Intenciones que difieren en un matiz temporal o de estado.** `pending_transfer` (13 errores, la peor) se va sobre
   todo a `balance_not_updated_after_bank_transfer` (6) y a `transfer_not_received_by_recipient` (4): «When will the
   transfer go through?» se predijo como no recibida. Y `balance_not_updated_after_bank_transfer` se va a
   `transfer_timing` con «When will my transfer be available in my account.». Pendiente, sin llegar y sin reflejarse en
   el saldo son la misma situación contada desde ángulos distintos.
2. **Pares casi sinónimos.** `why_verify_identity` ↔ `verify_my_identity` tiene la similitud más alta (0.30,
   percentil 99.9) y comparte seis bigramas frecuentes.
3. **Etiquetas dudosas.** «How do I top up my card?» está etiquetada como `transfer_into_account` y «How long can an EU
   transfer take?» como `pending_transfer`, cuando `topping_up_by_card` y `transfer_timing` (lo que predijo el modelo)
   son al menos igual de razonables. El ruido de anotación de BANKING77 está documentado (Ying y Thomas, 2022).

**`top_up_failed` ↔ `top_up_reverted`, el par con más errores (10), es un caso aparte:** comparten vocabulario
(similitud TF-IDF en el percentil 99.4) pero **no bigramas frecuentes**. La diferencia está en el verbo —*didn't go
through* / *not work* frente a *reverted*— y en la práctica el cliente describe lo mismo, así que el modelo no tiene una
pista léxica estable. `declined_transfer` es distinto otra vez: **precision 1.0 y recall 0.75**; sus 10 errores se
reparten entre `failed_transfer` (3), `declined_card_payment` (3) y otras cuatro clases, porque *declined* y *can't*
aparecen en varias intenciones de rechazo.
""",
"transformer": """
- **El transfer learning funciona muy bien con poco dato.** Con unos 117 ejemplos por clase y **8 minutos** de
  entrenamiento en un portátil (Apple Silicon, MPS), DistilRoBERTa alcanza **93.25 % de accuracy y 0.932 de F1 macro**
  en las 3 080 consultas de prueba, en línea con lo publicado para BANKING77 (≈ 93 % con BERT-base, de 12 capas,
  Casanueva et al., 2020) con la mitad de capas.
- **El entrenamiento es sano y reproducible.** La mejor época es la 6 (F1 macro de validación 0.930) y la métrica en
  prueba (0.932) coincide con la de validación: la validación era representativa. Al repetir la corrida completa se
  obtienen exactamente las mismas 3 080 predicciones.
- **El desempeño es desigual entre clases.** Nueve clases tienen F1 = 1.0 y cuatro quedan por debajo de 0.85, con el
  mínimo en `balance_not_updated_after_bank_transfer` (0.762). Las siete peores concentran **59 de los 208 errores
  (28 %)** siendo el 9 % de las clases. Las mejores tienen vocabulario exclusivo (*age*, *Apple Pay*, *ATM*,
  *terminate*, *passcode*); las peores lo comparten con sus vecinas (H2).
- **La confianza informa, pero no basta.** La celda de 3.4 lo mide: confianza media de **0.971** en los aciertos y
  **0.752** en los errores, pero **el 34.6 % de los errores (72 de 208) se comete con confianza superior a 0.90**. Un
  umbral de derivación a una persona atraparía parte de los errores, no todos.
- **Qué mejoraría el resultado:** ver la sección 3.8, donde cada mejora se deriva de un hallazgo medido.
""",
"mejoras": """
Cada mejora sale de un hallazgo medido en este notebook, no de una lista genérica:

| Hallazgo (sección) | Ajuste propuesto | Por qué | Referencia |
|---|---|---|---|
| El desbalance de train (5.3×) **no** muestra relación con el F1: ρ = −0.14, p = 0.21 (3.6, H1) | **No** priorizar *class weights*; si se prueban, pérdida ponderada por el número efectivo de muestras | Reponderar clases pequeñas que ya aciertan (F1 ≥ 0.886) movería la frontera a costa de las grandes sin atacar la causa | Cui et al. (2019) |
| Los pares confusos comparten 9× más bigramas frecuentes (3.6, H2) | ***Data augmentation* textual dirigida** a esos pares: paráfrasis que marquen el matiz (pendiente / no recibido / saldo sin actualizar) con *back-translation* y operaciones EDA | Aumentar datos al azar no ayuda; ampliar la frontera entre clases vecinas sí | Wei y Zou (2019); Sennrich et al. (2016) |
| Varias etiquetas del dataset son dudosas (3.6 y revisión manual de 4.7) | **Detectar etiquetas ruidosas** con *confident learning* y corregirlas antes de reentrenar | Parte del techo del modelo es ruido de anotación, no capacidad | Northcutt et al. (2021); Ying y Thomas (2022) |
| El 34.6 % de los errores se comete con confianza > 0.90 (3.4) | **Calibrar las probabilidades** con *temperature scaling* y derivar a una persona las consultas bajo un umbral | Sin calibrar, la confianza no sirve para abstenerse | Guo et al. (2017) |
| Las stopwords llevan la intención (2.2) | Mantener el **texto original** como entrada del Transformer (ya aplicado) | RoBERTa se preentrenó con texto sin limpiar; quitar *not* o *why* cambia la intención | Liu et al. (2019) |
| `top_up_failed` ↔ `top_up_reverted` se confunden sin compartir bigramas frecuentes (3.6) | Modelo de más capacidad (RoBERTa-base) o **ensamble** | Más capas de atención separan mejor matices que no dependen de expresiones compartidas | Liu et al. (2019); Sanh et al. (2019) |

**El vínculo EDA → ajustes, en una línea:** el análisis exploratorio señaló dos sospechosos —el desbalance y el
vocabulario compartido—; la evaluación no encontró relación con el primero y confirmó la asociación con el segundo,
así que la inversión va a los datos de los pares confusos y a la calidad de las etiquetas, no a reponderar clases.
""",
"calibracion_decodificacion": """
**Resultado (P1 sobre 8 consultas, 5 configuraciones):**

- **La temperatura alta es la que inventa historias.** C y D (T = 1.0) atribuyen el error a que el cliente «escribió
  mal» una palabra en las consultas 2, 3 y 6 (*«The customer probably mistyped…»*), algo que ninguna consulta dice. Lo
  hacen **con 60 y con 150 tokens por igual**, y ninguna de las configuraciones con T ≤ 0.3 lo hace: el efecto es de
  la temperatura, no de la longitud. Es el comportamiento esperable del muestreo con temperatura alta, que da
  probabilidad a continuaciones poco probables (Holtzman et al., 2020).
- **La longitud decide si la respuesta se corta.** Con la misma temperatura, B (60 tokens) deja 2 de 8 respuestas
  truncadas a media frase y E (150 tokens) ninguna; lo mismo pasa entre C (3/8 en formato correcto) y D (5/8). Las
  respuestas completas rara vez pasan de 60 tokens, pero algunas los necesitan.
- **Regla fijada de antemano:** gana **E** (T = 0.3, `top_p` = 0.9, 150 tokens) con **8 de 8** respuestas en formato
  correcto.

**Un matiz honesto:** con la estructura P1 ninguna configuración resuelve la **fidelidad** de las citas. E tiene 4 de
8 respuestas con cita falsa (por ejemplo, afirma que la consulta 1 contiene *card*, y no lo contiene) y la
codiciosa A, 2 (en la consulta 1 A sí cita *payment*, que está). La regla priorizó el formato; la fidelidad se
atacó en el paso siguiente, cambiando la estructura del prompt. Y la métrica automática tiene falsos positivos
cuando la puntuación queda dentro de las comillas (*«currency.»*), por lo que se usa solo para comparar
configuraciones, no como juicio final.
""",
"calibracion_estructura": """
**Resultado (configuración E fija, mismas 8 consultas):**

| | Citas verificables | Citas falsas | Formato correcto |
|---|---|---|---|
| **P1** · premisa de error | 1 | 4 | 8 |
| **P2** · neutral con citas | 6 | 2 | 8 |
| **P3** · P2 + dos ejemplos | **8** | **0** | 8 |

La **estructura** es lo que más mejora la fidelidad: pedir citas exactas (P2) obliga al modelo a anclarse en la
consulta, y los dos ejemplos resueltos (P3; *few-shot*, Brown et al., 2020) le enseñan el formato y el tipo de
razonamiento. La regla elige **P3**. Su coste es un prompt más largo (272 tokens frente a 110), que en este caso no
alarga la respuesta: P3 contesta con menos tokens de media que P1.
""",
"razones": """
**Veredicto de la revisión manual: 7 pertinentes, 10 parciales y 3 alucinadas de 20** (35 %, 50 % y 15 %). Con la
primera versión del prompt (P1 con T = 0.3 y 60 tokens), las mismas 20 consultas daban 3, 9 y 8: la calibración
completa **redujo las alucinaciones de 8 a 3**.

| Razón que da el LLM | Nº | Qué dice |
|---|---|---|
| **Solapamiento léxico** | 16 | «la palabra X de la consulta apunta a la clase predicha» — casi siempre cita una palabra real (*double charged*, *unblock*, *declined*, *PIN*) |
| **Ambigüedad real** | 2 | la consulta encaja en las dos clases (#10 *top-ups* sin especificar; #16 *disposable* en ambas) |
| **Etiqueta dudosa** | 2 | concluye por su cuenta que el clasificador acertó (#11 «How do I top up my card?»; #19 «How long can an EU transfer take?») |

Lo que sigue fallando, y cómo:

1. **Confundir cuál es la etiqueta y cuál la predicción** (#4 y #6, alucinadas; #5, #14 y #15, con la terminología
   mezclada). Con dos nombres de intención parecidos en el prompt, el modelo pierde cuál es cuál.
2. **Citar palabras que no están** (#12 *cash*; #13 *failed*; #20 *topping up*). La métrica automática detecta estos
   casos, pero también marca un falso positivo (#10: *top up* frente a *top-ups*).
3. **Añadir hechos** (#2: atribuye el doble cargo a un fraude, que la consulta no menciona).

La tabla cruzada de la celda anterior muestra el límite de la verificación automática: solo 1 de las 3 alucinadas
tiene una cita falsa detectable; las otras dos alucinan en el razonamiento, no en la cita. Por eso la validación
manual sigue siendo imprescindible.

**Etiquetas dudosas.** En al menos 6 de los 20 casos (#3, #7, #9, #11, #18 y #19) la predicción del clasificador es
tan razonable como la etiqueta del dataset o más. Con el prompt neutral (P3), el LLM lo dijo por sí mismo en dos
(#11 y #19); con el prompt que afirmaba «is WRONG» (P1) no lo había dicho en ninguno.
""",
"llm": """
- **La calibración funciona, y cada eje controla una cosa distinta.** La temperatura controla la *invención*: con
  T = 1.0 aparecen historias sobre el cliente que con T ≤ 0.3 no aparecen. La longitud (`max_new_tokens`) controla el
  *truncamiento*: 60 tokens cortan algunas respuestas, 150 no. La estructura controla la *fidelidad*: pedir citas
  exactas y dar dos ejemplos llevó las citas verificables de 1 a 8 de 8 y las falsas de 4 a 0.
- **Aun calibrado, un LLM de 7 B explica pero no garantiza verdad.** 7 de 20 explicaciones son plenamente
  pertinentes y 3 afirman algo falso. La fluidez sigue siendo el riesgo: todas suenan razonables. Falcon no ve los
  pesos ni las atenciones del clasificador, así que su explicación es una **conjetura *post hoc* a partir del texto**,
  no una explicación causal del modelo.
- **El encuadre del prompt importa.** Afirmar que la predicción era incorrecta (P1) llevaba al modelo a racionalizar;
  el prompt neutral le permitió señalar, en #11 y #19, que el clasificador probablemente acertó frente a una
  etiqueta dudosa.
- **Uso recomendado.** Como herramienta de **triaje** para un analista humano (hipótesis sobre grupos de errores, pares
  de clases confusas, sospechas de etiqueta dudosa) es útil y barato (≈ 3 s por explicación en un portátil). Como
  explicación final para un cliente o un auditor, no, salvo combinada con métodos que sí miran dentro del
  clasificador —SHAP (Lundberg y Lee, 2017) o gradientes integrados (Sundararajan et al., 2017)— y con verificación
  automática de las citas.
""",
"generales": """
1. **El Transformer resuelve bien la tarea.** DistilRoBERTa afinado alcanza **93.25 % de accuracy y 0.932 de F1
   macro** sobre 77 intenciones, al nivel de lo publicado con BERT-base (el doble de capas), con unos 117 ejemplos por
   clase y 8 minutos de entrenamiento, de forma reproducible. El EDA explicaba de antemano por qué hacía falta un modelo
   contextual: consultas cortas que comparten vocabulario y cuyo sentido lo deciden palabras que un preprocesado
   clásico eliminaría (*not*, *why*, *how*).
2. **Los errores tienen estructura, y el EDA la anticipaba a medias.** Se concentran en familias de intenciones con
   vocabulario compartido (68.3 % de los errores dentro de la misma familia; 9× más bigramas compartidos en los pares
   confusos; ρ = 0.28 entre similitud TF-IDF y confusión). El desbalance que también mostraba el EDA no mostró relación
   con el desempeño (ρ = −0.14, p = 0.21). El techo del modelo lo pone tanto la **calidad de las etiquetas** como la
   arquitectura.
3. **El LLM ayuda a pensar, no a explicar.** Con la calibración completa (T = 0.3, 150 tokens y un prompt neutral con
   citas y ejemplos), Falcon-7b-instruct pasó de 8 a 3 explicaciones alucinadas de 20, pero solo 7 fueron plenamente
   pertinentes. La calibración reduce la invención y el truncamiento; no convierte una conjetura en una explicación
   verificada. **La validación manual no es opcional.**
4. **El hallazgo más valioso lo dio la revisión humana:** descubrir que parte de los «errores» del clasificador son
   aciertos frente a etiquetas dudosas; el LLM solo lo notó cuando el prompt dejó de afirmar que la predicción era
   incorrecta.

**Ejemplos del experimento que lo sostienen:**

- *EDA → Transformer*: `verify_my_identity` y `why_verify_identity` comparten los bigramas *verify identity* e
  *identity check*, y el modelo las confundió 6 veces entre sí.
- *Transformer → calidad de las etiquetas*: «How do I top up my card?» está etiquetada como `transfer_into_account`; el
  modelo predijo `topping_up_by_card` con 98.9 % de confianza, y el LLM con el prompt neutral concluyó que la
  predicción era correcta (#11).
- *Calibración*: con T = 1.0 el LLM inventó que el cliente «escribió mal» *exchange rate*; con T = 0.3 no. Con P1 citó
  *card* en una consulta que no la contiene; con P3 citó *pending*, que sí está.
- *Limpieza*: «My card is not working» queda en «card working» al quitar stopwords; por eso el modelo recibe el texto
  original.

**Limitaciones.** Una sola semilla (la corrida se repitió y dio las mismas predicciones, pero no se midió la
variación entre semillas); la validación manual la hizo una sola persona; las 20 muestras son los errores **más
seguros** del modelo, no una muestra aleatoria, y concentran etiquetas dudosas, así que ni el 7/20 ni el 3/20 se pueden
extrapolar a los 208 errores; la calibración usó 8 consultas; la métrica automática de citas tiene falsos positivos;
y Falcon se ejecutó en `float16` en MPS, no cuantizado como sería en Colab, lo que puede variar sus salidas.
""",
}


nb = nbf.read(NB, as_version=4)
hechas = set()
for celda in nb.cells:
    if celda.cell_type != "markdown":
        continue
    marca = celda.metadata.get("interpretacion")
    if marca in TEXTOS:
        celda.source = TEXTOS[marca].strip("\n")
        hechas.add(marca)
pendientes = [c.metadata.get("interpretacion") for c in nb.cells
              if c.cell_type == "markdown" and "INTERPRETACION" in c.source]
nbf.write(nb, NB)
print(f"{len(hechas)} interpretaciones escritas · pendientes: {pendientes}")
