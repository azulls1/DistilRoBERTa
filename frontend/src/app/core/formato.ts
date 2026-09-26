const entero = new Intl.NumberFormat('es-MX');

/** 0.9312 → "93.12 %" */
export const pct = (x: number | null | undefined, decimales = 2) =>
  x == null ? '—' : `${(x * 100).toFixed(decimales)} %`;

/** 10003 → "10,003" */
export const num = (x: number | null | undefined) => (x == null ? '—' : entero.format(x));

/** 0.93123 → "0.931" */
export const dec = (x: number | null | undefined, decimales = 3) => (x == null ? '—' : x.toFixed(decimales));

/** card_arrival → "card arrival" */
export const legible = (etiqueta: string) => etiqueta.replaceAll('_', ' ').replace('?', '').toLowerCase();
