import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Encabezado } from '../componentes/encabezado';

import { Barras } from '../componentes/barras';
import { EstadoTarea } from '../core/modelos';
import { legible, pct } from '../core/formato';

const MAX = 512;
const EJEMPLOS = [
  'I still have not received my new card, it has been two weeks.',
  'Why was I charged a fee for withdrawing cash?',
  'How do I add money to my account using Apple Pay?',
  'My transfer to a friend has not arrived yet.',
  'Can I change my PIN at an ATM?',
];
// Palabras frecuentes en español para avisar que el modelo solo se entrenó con inglés
const ESPANOL = /\b(el|la|los|las|mi|tarjeta|cuenta|dinero|por qué|cómo|transferencia|que|de|no)\b/i;

@Component({
  selector: 'app-clasificar',
  imports: [Encabezado, Barras],
  template: `
    <app-encabezado titulo="Clasificar una consulta" icono="rayo" etiqueta="En vivo · Celery + Redis">
      <span entrada>Escribe una consulta bancaria en inglés. La petición se encola (Celery + Redis) y un worker la clasifica con el
        DistilRoBERTa afinado, en CPU.</span>
    </app-encabezado>

    <section class="panel">
      <form class="grid gap-3" (submit)="$event.preventDefault(); enviar()">
        <label for="texto" class="text-sm font-medium">Consulta</label>
        <textarea id="texto" class="campo min-h-24 resize-y" [maxLength]="MAX" required
                  placeholder="e.g. I still have not received my new card"
                  [value]="texto()" (input)="texto.set($any($event.target).value)"
                  [attr.aria-invalid]="invalido()" aria-describedby="ayuda"></textarea>
        <div id="ayuda" class="flex flex-wrap justify-between gap-2 text-xs text-tenue">
          <span>Entre 1 y {{ MAX }} caracteres.</span>
          <span class="tabular-nums" [class.text-mal]="texto().length > MAX">{{ texto().length }} / {{ MAX }}</span>
        </div>
        @if (enEspanol()) {
          <p class="rounded-lg bg-aviso-suave px-3 py-2 text-sm text-aviso">
            Parece español. El modelo se entrenó solo con consultas en inglés: la predicción puede no ser fiable.
          </p>
        }
        <div class="flex flex-wrap items-center gap-2">
          <button class="boton" type="submit" [disabled]="invalido() || procesando()">
            @if (procesando()) { <span class="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden="true"></span> Procesando… } @else { Clasificar }
          </button>
          <span class="text-xs text-tenue">o prueba:</span>
          @for (e of EJEMPLOS; track e) {
            <button type="button" class="rounded-full border border-borde px-2.5 py-1 text-xs hover:bg-borde/40" (click)="texto.set(e)">{{ e.slice(0, 28) }}…</button>
          }
        </div>
      </form>

      <div class="mt-5" aria-live="polite">
        @if (error()) {
          <p class="rounded-lg bg-mal-suave px-3 py-2 text-sm text-mal" role="alert">{{ error() }}</p>
        }
        @if (resultado(); as r) {
          <div class="grid gap-4 rounded-xl border border-borde p-4">
            <div class="flex flex-wrap items-baseline justify-between gap-2">
              <p class="text-lg"><span class="text-tenue">Intención:</span> <strong>{{ r.nombre_legible }}</strong> <span class="mono text-sm text-tenue">({{ r.clase }})</span></p>
              <span class="etiqueta" [class]="r.confianza < 0.5 ? 'bg-aviso-suave text-aviso' : 'bg-bien-suave text-bien'">
                {{ r.confianza < 0.5 ? 'Baja confianza' : 'Confianza' }} {{ pct(r.confianza, 1) }}
              </span>
            </div>
            <app-barras titulo="Cinco intenciones más probables" [max]="1" [datos]="top5()" />
            <p class="text-xs text-tenue">Inferencia en {{ r.duracion_ms }} ms dentro del worker.</p>
          </div>
        }
      </div>
    </section>

    <section class="panel mt-6">
      <div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 class="font-semibold">Tus consultas de esta sesión</h2>
        <p class="text-xs text-tenue">Solo las ves tú: no se muestran las consultas de otros visitantes.</p>
      </div>
      @if (historial().length) {
        <div class="overflow-x-auto">
          <table class="tabla min-w-[36rem]">
            <thead><tr><th>Consulta</th><th>Intención</th><th class="text-right">Confianza</th><th class="text-right">Cuándo</th></tr></thead>
            <tbody>
              @for (i of historial(); track i.cuando) {
                <tr class="animate-fadeIn">
                  <td>{{ i.texto }}</td>
                  <td class="mono">{{ i.clase }}</td>
                  <td class="text-right tabular-nums">{{ pct(i.confianza, 1) }}</td>
                  <td class="text-right text-xs text-tenue">{{ hora(i.cuando) }}</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      } @else {
        <p class="text-sm text-tenue">Aún no has clasificado ninguna consulta en esta sesión.</p>
      }
    </section>
  `,
})
export class ClasificarPagina {
  private readonly http = inject(HttpClient);
  private readonly destroy = inject(DestroyRef);
  protected readonly MAX = MAX;
  protected readonly EJEMPLOS = EJEMPLOS;
  protected readonly pct = pct;

  protected readonly texto = signal('');
  protected readonly procesando = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly resultado = signal<EstadoTarea['resultado'] | null>(null);
  /** Historial local de la sesión (máx. 15); no se consulta ni se expone el de otros visitantes. */
  protected readonly historial = signal<{ texto: string; clase: string; confianza: number; cuando: string }[]>([]);
  private enviado = '';

  protected readonly invalido = computed(() => !this.texto().trim() || this.texto().length > MAX);
  protected readonly enEspanol = computed(() => ESPANOL.test(this.texto()) || /[ñáéíóú¿¡]/i.test(this.texto()));
  protected readonly top5 = computed(() =>
    (this.resultado()?.top5 ?? []).map((t, i) => ({
      etiqueta: legible(t.clase), valor: t.prob, texto: pct(t.prob, 1), tono: i === 0 ? ('acento' as const) : ('tenue' as const),
    })),
  );

  private temporizador: ReturnType<typeof setTimeout> | undefined;

  constructor() {
    this.destroy.onDestroy(() => clearTimeout(this.temporizador));
  }

  protected enviar() {
    if (this.invalido() || this.procesando()) return;
    this.procesando.set(true);
    this.error.set(null);
    this.resultado.set(null);
    this.enviado = this.texto().trim();
    this.http.post<{ task_id: string }>('/api/clasificar', { texto: this.enviado }).subscribe({
      next: ({ task_id }) => this.sondear(task_id, Date.now()),
      error: (e) => this.fallar(e?.error?.detail ?? 'No se pudo enviar la consulta.'),
    });
  }

  private sondear(id: string, inicio: number) {
    this.http.get<EstadoTarea>(`/api/tareas/${id}`).subscribe({
      next: (t) => {
        if (t.estado === 'completada') {
          this.resultado.set(t.resultado ?? null);
          this.procesando.set(false);
          if (t.resultado) {
            const r = t.resultado;
            this.historial.update((h) => [{ texto: this.enviado, clase: r.clase, confianza: r.confianza, cuando: new Date().toISOString() }, ...h].slice(0, 15));
          }
        } else if (t.estado === 'error') {
          this.fallar(t.error ?? 'El worker no pudo clasificar la consulta.');
        } else if (Date.now() - inicio > 20000) {
          this.fallar('La clasificación tardó demasiado. Intenta de nuevo en un momento.');
        } else {
          this.temporizador = setTimeout(() => this.sondear(id, inicio), 500);
        }
      },
      error: () => this.fallar('Se perdió la conexión mientras se esperaba el resultado.'),
    });
  }

  private fallar(msg: string) {
    this.error.set(typeof msg === 'string' ? msg : 'Error inesperado.');
    this.procesando.set(false);
  }

  protected hora(iso: string) {
    return new Date(iso).toLocaleString('es-MX', { dateStyle: 'short', timeStyle: 'short' });
  }
}
