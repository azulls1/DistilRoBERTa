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
    texto: '¿Qué aprendimos del Transformer? Primero, que el aprendizaje por transferencia funciona muy bien con pocos datos: con unos 117 ejemplos por clase y ocho minutos de entrenamiento en un portátil, DistilRoBERTa llega al 93 % de exactitud, al nivel de lo publicado con BERT, que tiene el doble de capas. Segundo, que el entrenamiento fue sano: la mejor época fue la sexta; a partir de ahí el modelo empezaba a memorizar, así que nos quedamos con ella. Tercero, que los errores no son aleatorios. Nueve clases se clasifican perfecto, pero siete concentran más de una cuarta parte de los fallos, y casi siempre se confunden con su vecina más parecida: una transferencia pendiente con una que no ha llegado, o una recarga fallida con una revertida. Los pares que se confunden comparten nueve veces más expresiones que los que no. Y cuarto, que parte de esos errores no son del modelo, sino de etiquetas discutibles en el propio conjunto de datos.',
  },
  llm: {
    src: '/audio/llm.mp3',
    titulo: '¿Qué aprendimos del modelo de lenguaje?',
    texto: '¿Y qué aprendimos del modelo de lenguaje? Calibramos Falcon-7B en dos pasos. Primero, la decodificación: cinco configuraciones que separan la temperatura de la longitud. Con temperatura 1, el modelo inventaba historias, como que el cliente había escrito mal una palabra; y con sesenta tokens, algunas respuestas se cortaban a media frase. Ganó temperatura 0.3 con ciento cincuenta tokens. Después, la estructura del prompt: al pedirle que citara textualmente la consulta y darle dos ejemplos resueltos, las citas verificables pasaron de una a ocho de ocho, sin ninguna falsa. Al revisar a mano las veinte explicaciones finales, siete fueron plenamente pertinentes, diez parciales y tres afirmaban algo falso; con el primer prompt habían sido ocho. Lo más revelador: en varios casos el clasificador tenía razón y la etiqueta del conjunto era dudosa, y con el prompt neutral el propio modelo lo señaló en dos de ellos. La conclusión es clara: un modelo de lenguaje ayuda a pensar, pero no explica al clasificador por dentro. La validación humana no es opcional.',
  },
} as const;
