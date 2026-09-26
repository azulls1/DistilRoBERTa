import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { HttpClient, httpResource } from '@angular/common/http';
import { Encabezado } from '../componentes/encabezado';
import { Estado } from '../componentes/estado';
import { Icono, NombreIcono } from '../componentes/icono';
import { Barras } from '../componentes/barras';
import { Curva } from '../componentes/curva';
import { CalibracionCompleta, Epoca, EstadoTarea, Explicacion, Muestra, Resumen, SalidaLlm } from '../core/modelos';
import { num } from '../core/formato';
import { legible, pct } from '../core/formato';

type Resultado = NonNullable<EstadoTarea['resultado']>;
const PASOS: { titulo: string; icono: NombreIcono; detalle: string }[] = [
  { titulo: 'Consulta de entrada', icono: 'texto', detalle: 'El texto que escribe el cliente, en inglés.' },
  { titulo: 'Limpieza del análisis exploratorio', icono: 'lista', detalle: 'Minúsculas, sin caracteres especiales ni stopwords. Solo para el EDA: el Transformer recibe el texto original.' },
  { titulo: 'Tokenización BPE de DistilRoBERTa', icono: 'etiqueta', detalle: 'El tokenizador parte el texto en sub-palabras. «▁» marca un espacio antes del token.' },
  { titulo: 'Inferencia: 6 capas + clasificador', icono: 'capas', detalle: 'Autoatención en 6 capas, cabeza lineal de 77 salidas y softmax.' },
  { titulo: 'Veredicto frente a la etiqueta real', icono: 'diana', detalle: 'Solo en consultas del conjunto de prueba, que traen su etiqueta.' },
  { titulo: 'Explicación de Falcon-7b-instruct', icono: 'mensaje', detalle: 'Salida real generada offline con el prompt calibrado.' },
];
/** Descripción legible de una configuración a partir de sus parámetros reales (leídos de la base). */
function describir(p: Record<string, unknown> | undefined): string {
  if (!p) return '';
  const t = p['temperature'] == null ? 'codiciosa (T→0)' : `temperature=${p['temperature']} · top_p=${p['top_p']}`;
  return `${t} · max_new_tokens=${p['max_new_tokens']}`;
}

@Component({
  selector: 'app-simulacion',
  imports: [Encabezado, Estado, Icono, Barras, Curva],
  template: `
    <app-encabezado titulo="Simulación del sistema" icono="simulacion" etiqueta="C2 · C3 · en vivo">
      <span entrada>Recorre, paso a paso y con datos reales, lo que pasa con una consulta bancaria: la limpieza del EDA,
        los tokens que ve DistilRoBERTa, su predicción calculada en vivo por el worker de Celery y, si se equivoca,
        la explicación que dio Falcon-7b-instruct.</span>
    </app-encabezado>

    <!-- 1. Elegir la consulta -->
    <section class="panel">
      <div class="flex flex-wrap items-center gap-2" role="tablist" aria-label="Origen de la consulta">
        <button type="button" role="tab" [attr.aria-selected]="modo() === 'real'" (click)="modo.set('real')"
                class="rounded-full border px-3.5 py-1.5 text-sm transition" [class]="modo() === 'real' ? 'border-forest bg-forest text-white' : 'border-fog hover:bg-acento-suave'">
          Consulta real del conjunto de prueba</button>
        <button type="button" role="tab" [attr.aria-selected]="modo() === 'propia'" (click)="modo.set('propia')"
                class="rounded-full border px-3.5 py-1.5 text-sm transition" [class]="modo() === 'propia' ? 'border-forest bg-forest text-white' : 'border-fog hover:bg-acento-suave'">
          Escribir la mía</button>
      </div>

      @if (modo() === 'real') {
        <p class="mt-4 text-sm text-tenue">Toma al azar una de las {{ num(resumen.value()?.corrida?.n_test) }} consultas de prueba que el modelo nunca vio en el entrenamiento.</p>
        <div class="mt-3 flex flex-wrap gap-2">
          <button type="button" class="boton" [disabled]="corriendo()" (click)="simularReal('error')"><app-icono nombre="alerta" clase="h-4 w-4" /> Un error del modelo</button>
          <button type="button" class="boton" [disabled]="corriendo()" (click)="simularReal('acierto')"><app-icono nombre="checkCirculo" clase="h-4 w-4" /> Un acierto</button>
          <button type="button" class="boton !bg-white !text-forest ring-1 ring-fog" [disabled]="corriendo()" (click)="simularReal('aleatoria')"><app-icono nombre="dado" clase="h-4 w-4" /> Al azar</button>
        </div>
      } @else {
        <form class="mt-4 flex flex-col gap-3 sm:flex-row" (submit)="$event.preventDefault(); simularPropia()">
          <label class="sr-only" for="propia">Consulta</label>
          <input id="propia" class="campo flex-1" maxlength="512" placeholder="e.g. Why was my card payment declined at the shop?"
                 [value]="propia()" (input)="propia.set($any($event.target).value)" />
          <button class="boton" type="submit" [disabled]="corriendo() || !propia().trim()"><app-icono nombre="reproducir" clase="h-4 w-4" /> Simular</button>
        </form>
      }
      @if (error()) { <p class="mt-3 rounded-lg bg-mal-suave px-3 py-2 text-sm text-mal" role="alert">{{ error() }}</p> }
    </section>

    <!-- 2. Pasos -->
    @if (entrada()) {
      <ol class="relative mt-6 grid gap-4 before:absolute before:bottom-6 before:left-[1.35rem] before:top-6 before:w-px before:bg-fog" aria-live="polite">
        @for (p of pasos; track p.titulo; let i = $index) {
          @if (visibles() > i) {
            <li class="relative grid grid-cols-[2.75rem_1fr] gap-3 animate-fadeInUp">
              <span class="z-10 flex h-11 w-11 items-center justify-center rounded-full border-2 border-forest transition duration-500"
                    [class.bg-forest]="hecho(i)" [class.text-white]="hecho(i)" [class.bg-white]="!hecho(i)" [class.text-forest]="!hecho(i)">
                <app-icono [nombre]="p.icono" clase="h-5 w-5" />
              </span>
              <div class="panel !p-4">
                <p class="font-mono text-[10px] uppercase tracking-[0.14em] text-moss">Paso {{ i + 1 }}</p>
                <h2 class="font-display text-base font-semibold text-forest">{{ p.titulo }}</h2>
                <p class="mb-3 text-xs text-tenue">{{ p.detalle }}</p>

                @switch (i) {
                  @case (0) {
                    <blockquote class="rounded-lg bg-acento-suave px-4 py-3 text-[15px] text-forest">“{{ entrada() }}”</blockquote>
                    @if (muestra(); as m) { <p class="mt-2 text-xs text-tenue">Etiqueta real en el dataset: <span class="mono text-bien">{{ m.real }}</span> · consulta #{{ m.consulta_id - 100000 }} de prueba</p> }
                  }
                  @case (1) {
                    @if (resultado(); as r) {
                      <div class="flex flex-wrap gap-1.5">
                        @for (w of palabras(r.texto_limpio); track $index) { <span class="token-entra rounded-md bg-acento-suave px-2 py-0.5 font-mono text-xs text-evergreen" [style.animation-delay.ms]="$index * 40">{{ w }}</span> }
                        @empty { <span class="text-xs text-tenue">(sin palabras de contenido tras quitar stopwords)</span> }
                      </div>
                    }
                  }
                  @case (2) {
                    @if (resultado(); as r) {
                      <div class="flex flex-wrap gap-1">
                        @for (t of r.tokens; track $index) {
                          <span class="token-entra group relative rounded-md border px-1.5 py-0.5 font-mono text-xs"
                                [class]="t.token === '<s>' || t.token === '</s>' ? 'border-forest bg-forest text-white' : 'border-fog bg-white text-forest'"
                                [style.animation-delay.ms]="$index * 45" [title]="'id ' + t.id">{{ t.token }}</span>
                        }
                      </div>
                      <p class="mt-2 font-mono text-[11px] text-moss">{{ r.tokens.length }} tokens · ids: {{ ids(r) }}</p>
                    }
                  }
                  @case (3) {
                    @if (resultado(); as r) {
                      <div class="mb-4 grid grid-cols-7 gap-1.5" aria-hidden="true">
                        @for (c of capas; track c; let k = $index) {
                          <span class="capa-activa rounded-md bg-acento-suave py-2 text-center font-mono text-[10px] text-pine" [style.animation-delay.ms]="k * 110">{{ c }}</span>
                        }
                      </div>
                      <app-barras titulo="Cinco intenciones más probables" [max]="1" [datos]="top5(r)" />
                      <p class="mt-2 font-mono text-[11px] text-moss">Predicción: <span class="text-forest">{{ r.clase }}</span> · {{ pct(r.confianza, 1) }} · {{ r.duracion_ms }} ms en CPU (worker de Celery)</p>
                    }
                  }
                  @case (4) {
                    @if (muestra(); as m) {
                      @if (resultado(); as r) {
                        <div class="flex flex-wrap items-center gap-3">
                          @if (r.clase === m.real) {
                            <span class="etiqueta gap-1 bg-bien-suave !text-sm text-bien"><app-icono nombre="check" clase="h-4 w-4" /> Acierto</span>
                          } @else {
                            <span class="etiqueta gap-1 bg-mal-suave !text-sm text-mal"><app-icono nombre="x" clase="h-4 w-4" /> Error</span>
                          }
                          <span class="text-sm">Real <span class="mono text-bien">{{ m.real }}</span> · Predicha <span class="mono" [class]="r.clase === m.real ? 'text-bien' : 'text-mal'">{{ r.clase }}</span></span>
                        </div>
                        @if (r.clase !== m.pred_guardada) {
                          <p class="mt-2 text-xs text-aviso">En la evaluación del notebook (GPU de Apple, MPS) el modelo predijo <span class="mono">{{ m.pred_guardada }}</span>; en CPU el cálculo numérico cambia levemente y esta vez salió distinto.</p>
                        }
                      }
                    } @else {
                      <p class="text-sm text-tenue">Es una consulta tuya: no tiene etiqueta real con la cual comparar.</p>
                    }
                  }
                  @case (5) {
                    @if (muestra(); as m) {
                      @if (!m.correcta) {
                        <app-estado [cargando]="llm.isLoading()" [error]="llm.error()" [vacio]="!llm.value()?.length" (reintentar)="llm.reload()"
                                    textoVacio="No hay salida de Falcon para esta consulta.">
                          @if (salidaB(); as s) {
                            <p class="mb-2 text-xs text-tenue">Falcon explicó por qué el modelo eligió <span class="mono text-mal">{{ m.pred_guardada }}</span> en vez de <span class="mono text-bien">{{ m.real }}</span> (configuración {{ s.config }}, estructura {{ s.prompt }}, {{ s.segundos }} s en MPS):</p>
                            <blockquote class="border-l-2 border-forest pl-3 text-sm leading-relaxed text-forest">{{ s.explicacion }}</blockquote>
                            <p class="mt-2 text-xs" [class]="s.revisada ? 'text-pine' : 'text-aviso'">
                              {{ s.revisada ? 'Es una de las ' + nRevisadas() + ' explicaciones revisadas a mano: consulta su veredicto en «Explicaciones del LLM».' : 'Sin revisión manual: en la revisión de ' + nRevisadas() + ' casos, el ' + pctAlucinadas() + ' de las explicaciones de Falcon afirmaba algo falso. Léela con ese filtro.' }}
                            </p>
                          }
                        </app-estado>
                      } @else {
                        <p class="text-sm text-tenue">El modelo acertó en la evaluación: no hay error que explicar.</p>
                      }
                    } @else {
                      <p class="text-sm text-tenue">Falcon-7b (7 mil millones de parámetros, ≈ 14 GB en float16) no se sirve en vivo: el servidor no tiene GPU. Sus explicaciones se generaron offline para los {{ num(resumen.value()?.n_errores) }} errores del conjunto de prueba; elige «Un error del modelo» para ver una real.</p>
                    }
                  }
                }
              </div>
            </li>
          }
        }
      </ol>
    }

    <!-- 3. Simulador de temperatura -->
    <section class="mt-10">
      <h2 class="seccion flex items-center gap-2"><app-icono nombre="termometro" clase="h-4 w-4" /> Simulador de calibración del prompt</h2>
      <app-estado [cargando]="cal.isLoading()" [error]="cal.error()" [vacio]="!cal.value()?.length" (reintentar)="cal.reload()">
        <div class="panel grid gap-4">
          <p class="text-sm text-tenue">Elige uno de los {{ cal.value()?.length }} errores revisados y cambia la configuración de decodificación: todas las salidas son reales, generadas con la misma estructura de prompt y la misma semilla.</p>
          <div class="grid gap-3 md:grid-cols-[1fr_auto]">
            <label class="grid gap-1 text-sm"><span class="text-tenue">Error</span>
              <select class="campo" [value]="ordenCal()" (change)="ordenCal.set(+$any($event.target).value)">
                @for (c of cal.value() ?? []; track c.orden) { <option [value]="c.orden">{{ c.orden }}. {{ c.texto }}</option> }
              </select>
            </label>
            <div class="grid gap-1 text-sm"><span class="text-tenue">Configuración</span>
              <div class="flex rounded-lg border border-fog p-0.5" role="radiogroup" aria-label="Configuración">
                @for (k of claves(); track k) {
                  <button type="button" role="radio" [attr.aria-checked]="configActiva() === k" (click)="config.set(k)"
                          class="rounded-md px-3 py-1.5 font-mono text-xs transition" [class]="configActiva() === k ? 'bg-forest text-white' : 'text-pine hover:bg-acento-suave'">{{ k }}@if (k === elegida()) {✓}</button>
                }
              </div>
            </div>
          </div>
          @if (calActual(); as c) {
            <p class="font-mono text-[11px] text-moss">Config. {{ configActiva() }}{{ configActiva() === elegida() ? ' (elegida por la calibración)' : '' }} · {{ paramsConfig(configActiva()) }} · estructura {{ salidaCal()?.prompt }}</p>
            <div class="grid gap-3 md:grid-cols-2">
              <div class="rounded-lg bg-acento-suave p-3 text-xs leading-relaxed">
                <p class="mb-1 font-mono text-[10px] uppercase tracking-wider text-moss">Prompt enviado</p>
                <p>…Customer query: "<span class="text-forest">{{ c.texto }}</span>"<br />Predicted intent: <span class="text-mal">{{ leg(c.pred) }}</span><br />Correct intent: <span class="text-bien">{{ leg(c.real) }}</span><br /><br />Explanation:</p>
              </div>
              @if (salidaCal(); as s) {
                <div class="rounded-lg border border-fog p-3">
                  <p class="mb-1 flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-moss">Respuesta de Falcon
                    <span [class]="s.n_oraciones > 2 ? 'text-mal' : 'text-bien'">{{ s.n_oraciones }} oraciones · {{ s.segundos }} s</span></p>
                  <p class="animate-fadeIn text-sm leading-relaxed text-forest">{{ s.salida_cruda }}</p>
                  @if (configActiva() === elegida()) { <p class="mt-2 text-xs text-tenue">Veredicto manual: <strong>{{ c.veredicto }}</strong></p> }
                </div>
              }
            </div>
          }
        </div>
      </app-estado>
    </section>

    <!-- 4. Repetición del entrenamiento -->
    <section class="mt-10">
      <h2 class="seccion flex items-center gap-2"><app-icono nombre="reiniciar" clase="h-4 w-4" /> Repetición del entrenamiento</h2>
      <app-estado [cargando]="h.isLoading()" [error]="h.error()" (reintentar)="h.reload()">
        <div class="panel">
          <div class="mb-4 flex flex-wrap items-center gap-3">
            <button type="button" class="boton" (click)="reproducirEntrenamiento()" [disabled]="reproduciendo()">
              <app-icono [nombre]="reproduciendo() ? 'reloj' : 'reproducir'" clase="h-4 w-4" /> {{ reproduciendo() ? 'Entrenando…' : 'Reproducir las ' + (h.value()?.length ?? '') + ' épocas' }}</button>
            @if (epocaActual(); as e) {
              <span class="font-mono text-xs text-pine">Época {{ e.epoca }} · pérdida val. {{ e.eval_loss.toFixed(3) }} · accuracy val. {{ pct(e.eval_accuracy, 1) }}
                @if (e.epoca === mejorEpoca()) { <span class="ml-1 text-bien">← mejor época (early stopping)</span> }</span>
            }
          </div>
          <div class="max-w-3xl"><app-curva titulo="Pérdida por época" [epocas]="epocasVisibles()" [series]="seriesVisibles()" /></div>
        </div>
      </app-estado>
    </section>
  `,
})
export class SimulacionPagina {
  private readonly http = inject(HttpClient);
  private readonly temporizadores: ReturnType<typeof setTimeout>[] = [];
  protected readonly pasos = PASOS;
  protected readonly capas = ['capa 1', 'capa 2', 'capa 3', 'capa 4', 'capa 5', 'capa 6', 'softmax'];
  protected readonly claves = computed(() => [...new Set((this.cal.value() ?? []).flatMap((c) => c.configs.map((k) => k.config)))].sort());
  protected readonly pct = pct;
  protected readonly num = num;
  protected readonly leg = legible;
  protected readonly resumen = httpResource<Resumen>(() => '/api/resumen');
  private readonly exps = httpResource<Explicacion[]>(() => '/api/explicaciones');
  protected readonly nRevisadas = computed(() => this.exps.value()?.length ?? 0);
  protected readonly pctAlucinadas = computed(() => {
    const e = this.exps.value() ?? [];
    return e.length ? pct(e.filter((x) => x.veredicto === 'alucinada').length / e.length, 0) : '—';
  });
  protected readonly mejorEpoca = computed(() => Number(this.resumen.value()?.corrida?.hiperparametros['mejor_epoca'] ?? 0));

  protected readonly modo = signal<'real' | 'propia'>('real');
  protected readonly propia = signal('');
  protected readonly entrada = signal<string | null>(null);
  protected readonly muestra = signal<Muestra | null>(null);
  protected readonly resultado = signal<Resultado | null>(null);
  protected readonly visibles = signal(0);
  protected readonly corriendo = signal(false);
  protected readonly error = signal<string | null>(null);

  protected readonly llm = httpResource<SalidaLlm[]>(() => {
    const m = this.muestra();
    return m && !m.correcta && this.visibles() >= 6 ? `/api/simulacion/llm/${m.consulta_id}` : undefined;
  });
  protected readonly salidaB = computed(() => this.llm.value()?.find((s) => s.config === this.elegida()) ?? this.llm.value()?.[0] ?? null);

  protected readonly cal = httpResource<CalibracionCompleta[]>(() => '/api/simulacion/calibracion');
  protected readonly ordenCal = signal(1);
  protected readonly config = signal<string>('');
  /** Configuración que eligió la calibración del notebook (se muestra por defecto). */
  protected readonly elegida = computed(() => String(this.resumen.value()?.corrida?.hiperparametros['config_llm'] ?? ''));
  protected readonly configActiva = computed(() => this.config() || this.elegida());
  protected readonly calActual = computed(() => this.cal.value()?.find((c) => c.orden === this.ordenCal()) ?? null);
  protected readonly salidaCal = computed(() => this.calActual()?.configs.find((c) => c.config === this.configActiva()) ?? null);

  protected readonly h = httpResource<Epoca[]>(() => '/api/entrenamiento');
  protected readonly nEpocas = signal(8);
  protected readonly reproduciendo = signal(false);
  protected readonly epocasVisibles = computed(() => (this.h.value() ?? []).slice(0, this.nEpocas()).map((e) => e.epoca));
  protected readonly epocaActual = computed(() => (this.reproduciendo() ? (this.h.value() ?? [])[this.nEpocas() - 1] : null));
  protected readonly seriesVisibles = computed(() => {
    const h = (this.h.value() ?? []).slice(0, this.nEpocas());
    return [
      { nombre: 'train', valores: h.map((e) => e.train_loss), clase: 'text-forest' },
      { nombre: 'validación', valores: h.map((e) => e.eval_loss), clase: 'text-moss' },
    ];
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => this.temporizadores.forEach(clearTimeout));
  }

  protected simularReal(tipo: 'error' | 'acierto' | 'aleatoria') {
    this.reiniciar();
    this.http.get<Muestra>(`/api/simulacion/muestra?tipo=${tipo}`).subscribe({
      next: (m) => { this.muestra.set(m); this.iniciar(m.texto); },
      error: () => this.fallar('No se pudo obtener una consulta del conjunto de prueba.'),
    });
  }

  protected simularPropia() {
    this.reiniciar();
    this.iniciar(this.propia().trim());
  }

  private reiniciar() {
    this.temporizadores.forEach(clearTimeout);
    this.temporizadores.length = 0;
    this.corriendo.set(true);
    this.error.set(null);
    this.muestra.set(null);
    this.resultado.set(null);
    this.entrada.set(null);
    this.visibles.set(0);
  }

  private iniciar(texto: string) {
    this.entrada.set(texto);
    this.visibles.set(1);
    this.http.post<{ task_id: string }>('/api/clasificar', { texto }).subscribe({
      next: ({ task_id }) => this.sondear(task_id, Date.now()),
      error: (e) => this.fallar(e?.error?.detail ?? 'No se pudo enviar la consulta a la cola.'),
    });
  }

  private sondear(id: string, inicio: number) {
    this.http.get<EstadoTarea>(`/api/tareas/${id}`).subscribe({
      next: (t) => {
        if (t.estado === 'completada' && t.resultado) {
          this.resultado.set(t.resultado);
          // Los pasos se revelan uno tras otro para seguir el recorrido
          for (let k = 2; k <= PASOS.length; k++) {
            this.temporizadores.push(setTimeout(() => { this.visibles.set(k); if (k === PASOS.length) this.corriendo.set(false); }, (k - 1) * 700));
          }
        } else if (t.estado === 'error') {
          this.fallar(t.error ?? 'El worker no pudo clasificar la consulta.');
        } else if (Date.now() - inicio > 20000) {
          this.fallar('La clasificación tardó demasiado. Intenta de nuevo.');
        } else {
          this.temporizadores.push(setTimeout(() => this.sondear(id, inicio), 400));
        }
      },
      error: () => this.fallar('Se perdió la conexión con el servidor.'),
    });
  }

  private fallar(msg: string) {
    this.error.set(msg);
    this.corriendo.set(false);
  }

  protected reproducirEntrenamiento() {
    const total = this.h.value()?.length ?? 0;
    this.reproduciendo.set(true);
    for (let k = 1; k <= total; k++) {
      this.temporizadores.push(setTimeout(() => {
        this.nEpocas.set(k);
        if (k === total) this.temporizadores.push(setTimeout(() => this.reproduciendo.set(false), 1500));
      }, (k - 1) * 650 + 50));
    }
    this.nEpocas.set(1);
  }

  /** Un paso queda «hecho» cuando ya se reveló el siguiente (o es el último y todos están visibles). */
  protected hecho(i: number) {
    return this.visibles() > i + 1 || (i === PASOS.length - 1 && this.visibles() === PASOS.length);
  }
  protected palabras(t: string) {
    return t ? t.split(' ') : [];
  }
  protected ids(r: Resultado) {
    return r.tokens.map((t) => t.id).join(' ');
  }
  protected top5(r: Resultado) {
    return r.top5.map((t, i) => ({ etiqueta: legible(t.clase), valor: t.prob, texto: pct(t.prob, 1), tono: i === 0 ? ('acento' as const) : ('tenue' as const) }));
  }
  protected paramsConfig(k: string) {
    return describir(this.cal.value()?.[0] ? this.parametrosDe(k) : undefined);
  }
  /** Parámetros de una configuración, tomados de la tabla de calibración de la base. */
  private readonly calTabla = httpResource<{ config: string; parametros: Record<string, unknown> }[]>(() => '/api/calibracion');
  private parametrosDe(k: string) {
    return this.calTabla.value()?.find((c) => c.config === k)?.parametros;
  }

}
