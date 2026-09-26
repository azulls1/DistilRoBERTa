import { Component, ElementRef, afterRenderEffect, computed, signal, viewChild } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Confusion, ErrorPrediccion, Par } from '../core/modelos';
import { pct } from '../core/formato';

@Component({
  selector: 'app-confusion',
  imports: [Estado],
  template: `
    <header class="mb-6 grid gap-2">
      <h1 class="titulo">Matriz de confusión</h1>
      <p class="max-w-3xl text-tenue">
        77 × 77 celdas: filas = clase real, columnas = clase predicha. La diagonal (verde bosque) son aciertos; fuera de la
        diagonal (rojo), errores. Pasa el cursor o toca una celda para ver el detalle.
      </p>
    </header>

    <app-estado [cargando]="m.isLoading()" [error]="m.error()" [vacio]="!m.hasValue()" (reintentar)="m.reload()">
      <section class="grid gap-6 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div class="panel">
          <div class="relative mx-auto aspect-square w-full max-w-[40rem]">
            <canvas #lienzo class="h-full w-full cursor-crosshair rounded-md" role="img"
                    aria-label="Mapa de calor de la matriz de confusión; el detalle de la celda aparece al lado"
                    (mousemove)="apuntar($event)" (click)="apuntar($event)" (mouseleave)="celda.set(null)"></canvas>
          </div>
          <p class="mt-3 min-h-6 text-sm" aria-live="polite">
            @if (celda(); as c) {
              <span class="font-medium">{{ c.real }}</span> → <span class="font-medium">{{ c.pred }}</span>:
              <span class="mono">{{ c.conteo }}</span> {{ c.conteo === 1 ? 'consulta' : 'consultas' }}
              @if (c.real === c.pred) { <span class="text-bien">(aciertos)</span> }
            } @else {
              <span class="text-tenue">Sin celda seleccionada.</span>
            }
          </p>
        </div>

        <div class="panel">
          <h2 class="mb-3 font-semibold">Pares más confundidos</h2>
          <app-estado [cargando]="p.isLoading()" [error]="p.error()" (reintentar)="p.reload()">
            <ol class="grid gap-2 text-sm">
              @for (x of p.value() ?? []; track x.real + x.pred) {
                <li>
                  <button type="button" class="w-full rounded-lg px-2 py-1.5 text-left hover:bg-borde/40"
                          [class.bg-acento-suave]="claseErrores() === x.real" (click)="verErrores(x.real)">
                    <span class="mono text-mal">{{ x.conteo }}</span> {{ x.real }} <span class="text-tenue">→</span> {{ x.pred }}
                  </button>
                </li>
              }
            </ol>
          </app-estado>
        </div>
      </section>

      @if (claseErrores()) {
        <section class="panel mt-6">
          <div class="mb-3 flex items-center justify-between gap-3">
            <h2 class="font-semibold">Errores de «{{ claseErrores() }}»</h2>
            <button type="button" class="text-sm text-tenue hover:text-texto" (click)="claseErrores.set(null)">Cerrar</button>
          </div>
          <app-estado [cargando]="e.isLoading()" [error]="e.error()" [vacio]="!e.value()?.items?.length" (reintentar)="e.reload()">
            <div class="overflow-x-auto">
              <table class="tabla min-w-[36rem]">
                <thead><tr><th>Consulta</th><th>Predicha</th><th class="text-right">Confianza</th></tr></thead>
                <tbody>
                  @for (x of e.value()?.items ?? []; track x.consulta_id) {
                    <tr><td>{{ x.texto }}</td><td class="mono">{{ x.pred }}</td><td class="text-right tabular-nums">{{ pct(x.confianza, 1) }}</td></tr>
                  }
                </tbody>
              </table>
            </div>
          </app-estado>
        </section>
      }
    </app-estado>
  `,
})
export class ConfusionPagina {
  protected readonly m = httpResource<Confusion>(() => '/api/confusion');
  protected readonly p = httpResource<Par[]>(() => '/api/confusion/pares?limit=15');
  protected readonly claseErrores = signal<string | null>(null);
  protected readonly e = httpResource<{ total: number; items: ErrorPrediccion[] }>(() => {
    const clase = this.claseErrores();
    const id = clase ? this.m.value()?.clases.indexOf(clase) : -1;
    return id != null && id >= 0 ? `/api/errores?clase_real=${id}&limit=50` : undefined;
  });
  protected readonly celda = signal<{ real: string; pred: string; conteo: number } | null>(null);
  protected readonly pct = pct;

  private readonly lienzo = viewChild<ElementRef<HTMLCanvasElement>>('lienzo');
  private readonly matriz = computed(() => {
    const d = this.m.value();
    if (!d) return null;
    const n = d.clases.length;
    const x = new Uint16Array(n * n);
    for (const [i, j, c] of d.celdas) x[i * n + j] = c;
    return { n, x, clases: d.clases };
  });

  constructor() {
    afterRenderEffect(() => {
      const mat = this.matriz();
      const canvas = this.lienzo()?.nativeElement;
      if (!mat || !canvas) return;
      const lado = canvas.clientWidth;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.height = Math.round(lado * dpr);
      const ctx = canvas.getContext('2d')!;
      ctx.scale(dpr, dpr);
      const t = lado / mat.n;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, lado, lado);
      const maxFuera = Math.max(1, ...Array.from(mat.x).filter((_, k) => k % (mat.n + 1) !== 0));
      for (let i = 0; i < mat.n; i++) {
        for (let j = 0; j < mat.n; j++) {
          const c = mat.x[i * mat.n + j];
          if (!c) continue;
          if (i === j) {
            ctx.fillStyle = `rgba(4, 32, 44, ${0.25 + 0.75 * (c / 40)})`;
          } else {
            ctx.fillStyle = `rgba(220, 38, 38, ${0.25 + 0.75 * Math.sqrt(c / maxFuera)})`;
          }
          ctx.fillRect(j * t, i * t, Math.max(1, t - 0.5), Math.max(1, t - 0.5));
        }
      }
    });
  }

  protected apuntar(ev: MouseEvent) {
    const mat = this.matriz();
    const canvas = this.lienzo()?.nativeElement;
    if (!mat || !canvas) return;
    const r = canvas.getBoundingClientRect();
    const j = Math.floor(((ev.clientX - r.left) / r.width) * mat.n);
    const i = Math.floor(((ev.clientY - r.top) / r.height) * mat.n);
    if (i < 0 || j < 0 || i >= mat.n || j >= mat.n) return;
    this.celda.set({ real: mat.clases[i], pred: mat.clases[j], conteo: mat.x[i * mat.n + j] });
    if (ev.type === 'click' && i !== j) this.verErrores(mat.clases[i]);
  }

  protected verErrores(clase: string) {
    this.claseErrores.set(clase);
  }
}
