import { Component, computed, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Encabezado } from '../componentes/encabezado';
import { Narracion } from '../componentes/narracion';
import { NARRACIONES } from '../core/narraciones';

import { Kpi } from '../componentes/kpi';
import { Paginador, pagina } from '../componentes/paginador';
import { Calibracion, Explicacion } from '../core/modelos';
import { pct } from '../core/formato';

const VEREDICTO = {
  pertinente: { texto: 'Pertinente', clase: 'bg-bien-suave text-bien' },
  parcial: { texto: 'Parcial', clase: 'bg-aviso-suave text-aviso' },
  alucinada: { texto: 'Alucinada', clase: 'bg-mal-suave text-mal' },
} as const;

const RAZON: Record<string, string> = {
  solapamiento_lexico: 'Solapamiento léxico',
  ambiguedad_real: 'Ambigüedad real',
  etiqueta_dudosa: 'Etiqueta dudosa',
  generica: 'Genérica',
  frecuencia_supuesta: 'Frecuencia supuesta',
  otra: 'Otra',
};

/** Traducción de las líneas fijas de las plantillas P1–P3 (idénticas a las del notebook). */
const TRADUCCION: Record<string, string> = {
  'You are a banking customer-support analyst who audits an automatic intent classifier.':
    'Eres un analista de atención a clientes bancarios que audita un clasificador automático de intenciones.',
  'The classifier read the customer query below and predicted an intent that is WRONG.':
    'El clasificador leyó la consulta del cliente de abajo y predijo una intención que es INCORRECTA.',
  'In at most two short sentences, explain why the classifier probably chose the predicted intent instead of the correct one. Refer only to words that appear in the query. Do not invent facts, do not give advice to the customer.':
    'En máximo dos oraciones cortas, explica por qué el clasificador probablemente eligió la intención predicha en lugar de la correcta. Refiérete solo a palabras que aparecen en la consulta. No inventes hechos y no des consejos al cliente.',
  'The classifier read the customer query below and predicted an intent that differs from the label in the dataset.':
    'El clasificador leyó la consulta del cliente de abajo y predijo una intención distinta de la etiqueta del dataset.',
  "In at most two short sentences, explain which words of the query led the classifier to the predicted intent, quoting those words between single quotes exactly as they appear in the query. Do not invent facts, do not give advice to the customer.":
    'En máximo dos oraciones cortas, explica qué palabras de la consulta llevaron al clasificador a la intención predicha, citándolas entre comillas simples tal como aparecen en la consulta. No inventes hechos y no des consejos al cliente.',
  "Explanation: The query mentions 'top up' and a reverted top-up looks like a failed one, which pushes toward top up failed. The word 'reverted' is what points to the label.":
    "Explicación: La consulta menciona 'top up' y una recarga revertida se parece a una fallida, lo que empuja hacia top up failed. La palabra 'reverted' es la que apunta a la etiqueta.",
  "Explanation: The words 'wait' and 'my card' are typical of messages about cards that have not arrived. 'How long' asks for a delivery time, which fits the label better.":
    "Explicación: Las palabras 'wait' y 'my card' son típicas de mensajes sobre tarjetas que no han llegado. 'How long' pregunta por un plazo de entrega, que encaja mejor con la etiqueta.",
};
const PREFIJOS: [string, string][] = [
  ['Customer query: ', 'Consulta del cliente: '], ['Predicted intent: ', 'Intención predicha: '],
  ['Dataset label: ', 'Etiqueta del dataset: '], ['Correct intent: ', 'Intención correcta: '], ['Explanation:', 'Explicación:'],
];

@Component({
  selector: 'app-explicaciones',
  imports: [Encabezado, Narracion, Estado, Kpi, Paginador],
  template: `
    <app-encabezado titulo="Explicaciones del LLM" icono="mensaje" etiqueta="C3 · 3 pts · C4">
      <span entrada>Falcon-7b-instruct explica, en máximo dos oraciones, por qué el clasificador se equivocó en las {{ r.value()?.length }} consultas que
        falló con más confianza. Cada explicación se revisó a mano y lleva su veredicto.</span>
    </app-encabezado>
    <app-narracion class="mb-6 block animate-fadeInUp" [src]="narracion.src" [titulo]="narracion.titulo" [transcripcion]="narracion.texto" />

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.length" (reintentar)="r.reload()">
      <section class="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Resumen de la revisión manual">
        @for (k of conteo(); track k.etiqueta) {
          <app-kpi [etiqueta]="k.etiqueta" [valor]="k.valor" [detalle]="k.detalle" />
        }
      </section>

      <!-- Pestañas: evitan una página de 14 pantallas de alto -->
      <div class="mt-8 flex gap-1 overflow-x-auto rounded-xl border border-fog bg-white p-1" role="tablist" aria-label="Secciones">
        @for (t of pestanas; track t.id) {
          <button type="button" role="tab" [id]="'tab-' + t.id" [attr.aria-selected]="pestana() === t.id" [attr.aria-controls]="'panel-' + t.id"
                  class="flex shrink-0 items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition"
                  [class]="pestana() === t.id ? 'bg-forest text-white shadow-sm' : 'text-pine hover:bg-acento-suave'" (click)="pestana.set(t.id)">
            {{ t.texto }} <span class="rounded-full px-1.5 font-mono text-[10px]" [class]="pestana() === t.id ? 'bg-white/20' : 'bg-acento-suave'">{{ t.id === 'explicaciones' ? r.value()?.length : t.id === 'calibracion' ? nConfigs() : 'EN·ES' }}</span>
          </button>
        }
      </div>

      @if (pestana() === 'explicaciones') {
        <section id="panel-explicaciones" role="tabpanel" aria-labelledby="tab-explicaciones" class="animate-fadeIn scroll-mt-24">
          <div class="mt-4 flex flex-wrap gap-2" role="group" aria-label="Filtrar por veredicto">
            @for (f of filtros; track f.id) {
              <button type="button" class="rounded-full border px-3 py-1 text-sm transition"
                      [class]="filtro() === f.id ? 'border-acento bg-acento-suave text-acento' : 'border-borde hover:bg-borde/40'"
                      [attr.aria-pressed]="filtro() === f.id" (click)="filtro.set(f.id); pag.set(1)">{{ f.texto }} <span class="font-mono text-[10px] text-moss">{{ cuenta(f.id) }}</span></button>
            }
          </div>

          <div class="mt-4 grid gap-4 lg:grid-cols-2">
            @for (e of paginaExp(); track e.orden) {
              <article class="panel grid content-start gap-3 animate-fadeInUp">
                <div class="flex items-start justify-between gap-3">
                  <p class="font-medium"><span class="mr-1 font-mono text-xs text-moss">#{{ e.orden }}</span>“{{ e.texto }}”</p>
                  <span class="etiqueta shrink-0" [class]="veredicto(e.veredicto).clase">{{ veredicto(e.veredicto).texto }}</span>
                </div>
                <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
                  <dt class="text-tenue">Real</dt><dd class="mono break-all text-bien">{{ e.real }}</dd>
                  <dt class="text-tenue">Predicha</dt><dd class="mono break-all text-mal">{{ e.pred }} <span class="text-tenue">({{ pct(e.confianza, 1) }})</span></dd>
                  <dt class="text-tenue">Razón</dt><dd>{{ razon(e.razon_categoria) }}</dd>
                </dl>
                <blockquote class="border-l-2 border-acento pl-3 text-sm leading-relaxed">{{ e.explicacion }}</blockquote>
                @if (e.nota_revision) {
                  <details class="group text-xs text-tenue">
                    <summary class="cursor-pointer select-none font-medium text-pine hover:underline">Nota de la revisión manual</summary>
                    <p class="mt-1 leading-relaxed">{{ e.nota_revision }}</p>
                  </details>
                }
              </article>
            } @empty {
              <p class="text-sm text-tenue">No hay explicaciones con ese veredicto.</p>
            }
          </div>
          <app-paginador [total]="visibles().length" [(pagina)]="pag" [(tamano)]="tam" [opciones]="[6, 10, 20]" etiqueta="explicaciones" ancla="panel-explicaciones" />
        </section>
      }

      @if (pestana() === 'calibracion') {
        <section id="panel-calibracion" role="tabpanel" aria-labelledby="tab-calibracion" class="panel mt-4 animate-fadeIn scroll-mt-24">
          <h2 class="mb-1 font-semibold">Calibración del prompt</h2>
          <p class="subtitulo mb-4">
            {{ nConfigs() }} combinaciones de decodificación y estructura sobre las mismas {{ nConsultasCal() }} consultas; los parámetros salen
            de la base. Se marca la combinación que eligió la regla del notebook.
          </p>
          <div class="mb-4 flex flex-wrap gap-2" role="group" aria-label="Paso de la calibración">
            @for (p of pasos(); track p.id) {
              <button type="button" class="rounded-full border px-3 py-1 text-sm transition"
                      [class]="paso() === p.id ? 'border-acento bg-acento-suave text-acento' : 'border-borde hover:bg-borde/40'"
                      [attr.aria-pressed]="paso() === p.id" (click)="paso.set(p.id); pagCal.set(1)">{{ p.texto }}</button>
            }
          </div>

          <!-- Resumen por combinación: lo que decide la regla -->
          <div class="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
            @for (g of resumenPaso(); track g.clave) {
              <div class="rounded-lg border p-3 text-xs" [class]="g.elegida ? 'border-forest bg-acento-suave' : 'border-fog'">
                <p class="flex items-center justify-between font-mono text-sm font-semibold text-forest">{{ g.clave }} @if (g.elegida) { <span class="etiqueta bg-forest text-white">elegida</span> }</p>
                <p class="mt-0.5 font-mono text-[10px] text-moss">{{ g.params }}</p>
                <dl class="mt-2 grid grid-cols-[1fr_auto] gap-y-0.5 tabular-nums">
                  <dt class="text-tenue">Formato correcto</dt><dd class="font-medium">{{ g.formato }}/{{ g.n }}</dd>
                  <dt class="text-tenue">Citas verificables</dt><dd class="font-medium text-bien">{{ g.verif }}</dd>
                  <dt class="text-tenue">Citas falsas</dt><dd class="font-medium" [class.text-mal]="g.falsas">{{ g.falsas }}</dd>
                </dl>
              </div>
            }
          </div>

          <app-estado [cargando]="c.isLoading()" [error]="c.error()" (reintentar)="c.reload()">
            <div id="tabla-calibracion" class="mt-5 overflow-x-auto scroll-mt-24">
              <table class="tabla min-w-[48rem]">
                <thead><tr><th>Consulta</th><th>Config.</th><th class="text-right">Oraciones</th><th class="text-right">Tokens</th>
                  <th>Formato</th><th>Citas</th><th>Salida</th></tr></thead>
                <tbody>
                  @for (k of paginaCal(); track $index) {
                    <tr [class.bg-acento-suave]="k.elegida">
                      <td class="max-w-[14rem]">{{ k.consulta }}</td>
                      <td class="mono" [title]="parametros(k)">{{ k.config }}·{{ k.prompt }}@if (k.elegida) { ✓ }<span class="block text-[10px] text-moss">{{ parametros(k) }}</span></td>
                      <td class="text-right tabular-nums" [class.text-mal]="k.n_oraciones > 2">{{ k.n_oraciones }}</td>
                      <td class="text-right tabular-nums">{{ k.n_tokens }}</td>
                      <td [class]="k.formato_ok ? 'text-bien' : 'text-mal'">{{ k.formato_ok ? 'correcto' : 'no' }}</td>
                      <td [class]="k.cita_falsa ? 'text-mal' : k.cita_verificable ? 'text-bien' : 'text-tenue'">{{ k.cita_falsa ? 'falsa' : k.cita_verificable ? 'verificable' : '—' }}</td>
                      <td class="text-xs leading-relaxed">{{ k.salida }}</td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
            <app-paginador [total]="filasPaso().length" [(pagina)]="pagCal" [(tamano)]="tamCal" [opciones]="[10, 20, 40]" etiqueta="salidas" ancla="tabla-calibracion" />
          </app-estado>
        </section>
      }

      @if (pestana() === 'prompt') {
        @if (promptActual(); as e) {
          <section id="panel-prompt" role="tabpanel" aria-labelledby="tab-prompt" class="panel mt-4 animate-fadeIn">
            <div class="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 class="font-semibold">El prompt, en inglés y en español</h2>
                <p class="subtitulo">Se envía en inglés (el idioma del dataset y de las instrucciones con que se ajustó Falcon); la columna derecha
                  es su traducción línea por línea. Estructura {{ estructura() }}, configuración {{ configElegida() }}.</p>
              </div>
              <label class="grid w-full gap-1 text-sm sm:w-72"><span class="text-tenue">Error</span>
                <select class="campo w-full" [value]="ordenPrompt()" (change)="ordenPrompt.set(+$any($event.target).value)">
                  @for (x of r.value() ?? []; track x.orden) { <option [value]="x.orden">{{ x.orden }}. {{ x.texto }}</option> }
                </select>
              </label>
            </div>
            <div class="grid gap-4 lg:grid-cols-2">
              <div class="min-w-0">
                <p class="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-moss">Inglés · exacto, leído de la base</p>
                <pre class="mono h-full max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-lg bg-acento-suave p-4 text-xs leading-relaxed text-forest">{{ e.prompt }}</pre>
              </div>
              <div class="min-w-0">
                <p class="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-moss">Español · traducción</p>
                <pre class="mono h-full max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-lg border border-fog p-4 text-xs leading-relaxed text-evergreen">{{ traducir(e.prompt) }}</pre>
              </div>
            </div>
          </section>
        }
      }
    </app-estado>
  `,
})
export class ExplicacionesPagina {
  protected readonly narracion = NARRACIONES.llm;
  protected readonly r = httpResource<Explicacion[]>(() => '/api/explicaciones');
  protected readonly c = httpResource<Calibracion[]>(() => '/api/calibracion');
  protected readonly pct = pct;
  protected readonly ordenPrompt = signal(1);
  protected readonly promptActual = computed(() => this.r.value()?.find((e) => e.orden === this.ordenPrompt()) ?? null);
  /** Traducción al español, línea por línea, del prompt que realmente se envió (cualquier estructura P1–P3).
   *  Las consultas y los nombres de intención quedan en inglés porque son el dato original. */
  protected traducir(prompt: string) {
    return prompt.split('\n').map((l) => {
      if (TRADUCCION[l.trim()] !== undefined) return TRADUCCION[l.trim()];
      for (const [en, es] of PREFIJOS) if (l.startsWith(en)) return es + l.slice(en.length);
      return l;
    }).join('\n');
  }
  protected readonly estructura = computed(() => this.c.value()?.find((k) => k.elegida)?.prompt ?? '');
  protected readonly configElegida = computed(() => this.c.value()?.find((k) => k.elegida)?.config ?? '');
  protected readonly pestana = signal<'explicaciones' | 'calibracion' | 'prompt'>('explicaciones');
  protected readonly pestanas = [
    { id: 'explicaciones' as const, texto: 'Las explicaciones' },
    { id: 'calibracion' as const, texto: 'Calibración' },
    { id: 'prompt' as const, texto: 'Prompt' },
  ];
  protected readonly pag = signal(1);
  protected readonly tam = signal(6);
  protected readonly pagCal = signal(1);
  protected readonly tamCal = signal(10);
  /** Paso 1: decodificación con la estructura base; paso 2: estructuras con la configuración elegida. */
  protected readonly paso = signal<'decodificacion' | 'estructura'>('decodificacion');
  protected readonly pasos = computed(() => {
    const est = this.estructuras();
    return [
      { id: 'decodificacion' as const, texto: `Paso 1 · decodificación (${this.configs().join('–')}) con ${est[0] ?? ''}` },
      { id: 'estructura' as const, texto: `Paso 2 · estructura (${est.join('–')}) con ${this.configElegida()}` },
    ];
  });
  private readonly configs = computed(() => [...new Set((this.c.value() ?? []).map((k) => k.config))].sort());
  private readonly estructuras = computed(() => [...new Set((this.c.value() ?? []).map((k) => k.prompt))].sort());
  protected readonly filasPaso = computed(() => {
    const todas = this.c.value() ?? [];
    return this.paso() === 'decodificacion'
      ? todas.filter((k) => k.prompt === this.estructuras()[0])
      : todas.filter((k) => k.config === this.configElegida());
  });
  protected readonly paginaCal = computed(() => pagina(this.filasPaso(), this.pagCal(), this.tamCal()));
  protected readonly resumenPaso = computed(() => {
    const porClave = this.paso() === 'decodificacion' ? (k: Calibracion) => k.config : (k: Calibracion) => k.prompt;
    const grupos = new Map<string, Calibracion[]>();
    for (const k of this.filasPaso()) grupos.set(porClave(k), [...(grupos.get(porClave(k)) ?? []), k]);
    return [...grupos.entries()].sort().map(([clave, g]) => ({
      clave, n: g.length, elegida: g.some((k) => k.elegida),
      params: this.paso() === 'decodificacion' ? this.parametros(g[0]).split(' · ').slice(0, 2).join(' · ') : 'config. ' + g[0].config,
      formato: g.filter((k) => k.formato_ok).length, verif: g.filter((k) => k.cita_verificable).length,
      falsas: g.filter((k) => k.cita_falsa).length,
    }));
  });
  protected readonly nConfigs = computed(() => new Set((this.c.value() ?? []).map((k) => k.config + k.prompt)).size);
  protected readonly nConsultasCal = computed(() => new Set((this.c.value() ?? []).map((k) => k.consulta)).size);
  protected parametros(k: Calibracion) {
    const p = k.parametros;
    return (p['temperature'] == null ? 'T→0' : 'T=' + p['temperature']) + ' · ' + p['max_new_tokens'] + ' tok · ' + k.prompt;
  }
  protected readonly filtro = signal<'todas' | Explicacion['veredicto']>('todas');
  protected readonly filtros = [
    { id: 'todas' as const, texto: 'Todas' },
    { id: 'pertinente' as const, texto: 'Pertinentes' },
    { id: 'parcial' as const, texto: 'Parciales' },
    { id: 'alucinada' as const, texto: 'Alucinadas' },
  ];

  protected readonly visibles = computed(() => {
    const f = this.filtro();
    return (this.r.value() ?? []).filter((e) => f === 'todas' || e.veredicto === f);
  });
  protected readonly paginaExp = computed(() => pagina(this.visibles(), this.pag(), this.tam()));
  protected cuenta(f: 'todas' | Explicacion['veredicto']) {
    return (this.r.value() ?? []).filter((e) => f === 'todas' || e.veredicto === f).length;
  }

  protected readonly conteo = computed(() => {
    const e = this.r.value() ?? [];
    const n = (v: string) => e.filter((x) => x.veredicto === v).length;
    const recortadas = e.filter((x) => x.explicacion.trim() !== x.salida_cruda.trim()).length;
    const dos = e.filter((x) => (x.explicacion.match(/[.!?](\s|$)/g) ?? []).length <= 2).length;
    return [
      { etiqueta: 'Pertinentes', valor: `${n('pertinente')} / ${e.length}`, detalle: 'razón plausible y fiel a la consulta' },
      { etiqueta: 'Parciales', valor: `${n('parcial')} / ${e.length}`, detalle: 'algo cierto pero vago' },
      { etiqueta: 'Alucinadas', valor: `${n('alucinada')} / ${e.length}`, detalle: 'afirma lo que la consulta no dice' },
      { etiqueta: '≤ 2 oraciones', valor: `${dos} / ${e.length}`, detalle: recortadas ? `${recortadas} recortadas en el postproceso` : 'salida cruda, sin recortes' },
    ];
  });

  protected veredicto(v: Explicacion['veredicto']) {
    return VEREDICTO[v] ?? VEREDICTO.parcial;
  }
  protected razon(r: string) {
    return RAZON[r] ?? r;
  }
}
