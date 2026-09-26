/** Narraciones generadas con ElevenLabs (voz «David García», es-MX, eleven_multilingual_v2) y su transcripción. */
export const NARRACIONES = {
  recorrido: {
    src: '/audio/recorrido.mp3',
    titulo: 'Recorrido guiado de la actividad',
    texto: 'Bienvenido. Este portal presenta la Actividad dos de Sistemas Cognitivos Artificiales: Transformers y modelos de lenguaje grande. El reto es leer una consulta bancaria escrita por un cliente, en inglés, y decidir cuál de setenta y siete intenciones expresa: por ejemplo, que su tarjeta no ha llegado, o que le cobraron una comisión. Para resolverlo afinamos DistilRoBERTa, un Transformer de seis capas, con diez mil consultas etiquetadas del conjunto Banking77. En tres mil ochenta consultas que el modelo nunca vio, acierta el 93.2 %. Después usamos Falcon-7B, un modelo de lenguaje grande, para que explicara en dos oraciones por qué el clasificador se equivocó en veinte casos, y revisamos cada explicación a mano. En el menú puedes recorrer el análisis exploratorio, el desempeño por clase, la matriz de confusión y las explicaciones; en la simulación, ver paso a paso cómo se clasifica una consulta real; y en entregables, descargar el notebook y el PDF de la actividad.',
  },
  transformer: {
    src: '/audio/transformer.mp3',
    titulo: '¿Qué aprendimos del Transformer?',
    texto: '¿Qué aprendimos del Transformer? Primero, que el aprendizaje por transferencia funciona muy bien con pocos datos: con unos 117 ejemplos por clase y ocho minutos de entrenamiento en un portátil, DistilRoBERTa llega al 93 % de exactitud, al nivel de modelos del doble de tamaño. Segundo, que el entrenamiento fue sano: la mejor época fue la sexta, y a partir de ahí el modelo empezaba a memorizar, así que nos quedamos con ella. Tercero, que los errores no son aleatorios. Nueve clases se clasifican perfecto, pero siete concentran casi un tercio de los fallos, y casi siempre se confunden con su vecina más parecida: una transferencia pendiente con una que no ha llegado, o una recarga fallida con una revertida. Cuanto más vocabulario comparten dos intenciones, más se confunden. Y cuarto, que parte de esos errores no son del modelo, sino de etiquetas discutibles en el propio conjunto de datos.',
  },
  llm: {
    src: '/audio/llm.mp3',
    titulo: '¿Qué aprendimos del modelo de lenguaje?',
    texto: '¿Y qué aprendimos del modelo de lenguaje? Calibramos el prompt de Falcon-7B con tres configuraciones. Con temperatura 1, el modelo inventaba historias, como que el cliente había escrito mal una palabra. Con temperatura 0.3 y un límite de sesenta tokens, respetó el formato de dos oraciones en todos los casos. Pero el formato no es la verdad. Al revisar a mano las veinte explicaciones, solo tres fueron plenamente pertinentes, nueve fueron vagas y ocho afirmaban algo falso: citaban palabras que no estaban en la consulta, confundían la clase real con la predicha, o inventaban que una clase era más común. Lo más revelador: en varios casos el clasificador tenía razón y la etiqueta del conjunto era dudosa, pero el modelo de lenguaje nunca lo detectó, porque el prompt le decía que la predicción era incorrecta. La conclusión es clara: un modelo de lenguaje ayuda a pensar, pero no explica al clasificador. La validación humana no es opcional.',
  },
} as const;
