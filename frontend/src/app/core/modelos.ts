/** Tipos del contrato REST (espesificaciones/.../contracts/api.md). */

export interface Corrida {
  id: string;
  modelo_base: string;
  llm: string;
  dispositivo: string;
  hiperparametros: Record<string, number | string | null>;
  accuracy: number;
  f1_macro: number;
  n_train: number;
  n_val: number;
  n_test: number;
  duracion_entrenamiento_s: number | null;
  creado_en: string;
}

export interface Resumen {
  corrida: Corrida;
  n_clases: number;
  n_errores: number;
}

export interface Estadistica {
  metrica: string;
  particion: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q1: number;
  mediana: number;
  q3: number;
  max: number;
}

export interface Ngrama {
  rango: number;
  ngrama: string;
  frecuencia: number;
}

export interface Eda {
  estadisticas: Estadistica[];
  ngramas: { palabra: Ngrama[]; bigrama: Ngrama[]; trigrama: Ngrama[] };
  balance: Record<string, number | string>;
}

export interface Clase {
  id: number;
  nombre: string;
  nombre_legible: string;
  n_train: number;
  n_test: number;
  precision: number;
  recall: number;
  f1: number;
  soporte: number;
  errores: number;
  categoria: 'mejor' | 'peor' | null;
}

export interface Epoca {
  epoca: number;
  train_loss: number;
  eval_loss: number;
  eval_accuracy: number;
  eval_f1_macro: number;
}

export interface Confusion {
  clases: string[];
  celdas: [number, number, number][];
}

export interface Par {
  real: string;
  pred: string;
  conteo: number;
}

export interface ErrorPrediccion {
  consulta_id: number;
  texto: string;
  real: string;
  pred: string;
  confianza: number;
}

export interface Explicacion {
  orden: number;
  texto: string;
  real: string;
  pred: string;
  confianza: number;
  explicacion: string;
  salida_cruda: string;
  razon_categoria: string;
  veredicto: 'pertinente' | 'parcial' | 'alucinada';
  nota_revision: string | null;
  parametros: Record<string, number | null>;
  prompt: string;
}

export interface Calibracion {
  config: string;
  parametros: Record<string, number | null>;
  consulta: string;
  salida: string;
  n_oraciones: number;
  n_tokens: number;
  segundos: number;
  palabras_ajenas: number | null;
  elegida: boolean;
}

export interface Top5 {
  clase_id?: number;
  clase: string;
  prob: number;
}

export interface EstadoTarea {
  estado: 'pendiente' | 'completada' | 'error';
  resultado?: { clase: string; nombre_legible: string; confianza: number; top5: Top5[]; duracion_ms: number };
  error?: string;
}

export interface Inferencia {
  id: string;
  texto: string;
  clase: string | null;
  confianza: number | null;
  estado: string;
  creado_en: string;
}
