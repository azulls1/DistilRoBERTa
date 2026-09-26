import { Component, DestroyRef, ElementRef, afterRenderEffect, computed, inject, signal, viewChild } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Encabezado } from '../componentes/encabezado';
import { Icono } from '../componentes/icono';
import { Confusion, ErrorPrediccion, Par } from '../core/modelos';
import { pct } from '../core/formato';

interface Celda { i: number; j: number; c: number }
const DURACION_DIAGONAL = 900;   // ms: la diagonal se traza de esquina a esquina
const DURACION_ERRORES = 1100;   // ms: los errores aparecen del más leve al más grave
const sinMovimiento = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const suave = (x: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3);

@Component({
  selector: 'app-confusion',
  imports: [Encabezado, Estado, Icono],
  template: `
    <app-encabezado titulo="Matriz de confusión" icono="cuadricula" etiqueta="C2 · análisis de causas">
      <span entrada>{{ n() }} × {{ n() }} celdas: filas = clase real, columnas = clase predicha. La diagonal (verde bosque)
        son aciertos; fuera de la diagonal (rojo), errores. Pasa el cursor o toca una celda para ver el detalle; los
        tres pares más confundidos laten.</span>
    </app-encabezado>

    <app-estado [cargando]="m.isLoading()" [error]="m.error()" [vacio]="!m.hasValue()" (reintentar)="m.reload()">
      <section class="grid gap-6 xl:grid-cols-[minmax(0,1fr)_21rem]">
        <div class="panel">
          <div class="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div class="flex flex-wrap gap-4 font-mono text-[11px] text-moss">
              <span class="inline-flex items-center gap-1.5"><span class="h-3 w-3 rounded-sm bg-forest"></span> aciertos ({{ aciertos() }})</span>
              <span class="inline-flex items-center gap-1.5"><span class="h-3 w-3 rounded-sm bg-mal"></span> errores ({{ errores() }})</span>
              <span>{{ celdasError() }} pares distintos</span>
            </div>
            <button type="button" (click)="repetir()" class="inline-flex items-center gap-1.5 rounded-lg border border-fog px-3 py-1.5 text-xs text-forest transition hover:border-forest">
              <app-icono nombre="reiniciar" clase="h-3.5 w-3.5" /> Repetir animación</button>
          </div>

          <div class="relative mx-auto aspect-square w-full max-w-[40rem]" #marco>
            <canvas #lienzo class="h-full w-full cursor-crosshair rounded-md" role="img"
                    aria-label="Mapa de calor animado de la matriz de confusión; el detalle de la celda aparece debajo"
                    (mousemove)="apuntar($event)" (click)="apuntar($event)" (mouseleave)="salir()"></canvas>
            @if (celda(); as c) {
              <div class="pointer-events-none absolute z-10 w-max max-w-[16rem] -translate-x-1/2 rounded-lg bg-forest px-3 py-2 text-xs text-white shadow-[0_8px_24px_-6px_rgba(4,32,44,0.45)] transition-[left,top] duration-75"
                   [style.left.px]="tip().x" [style.top.px]="tip().y" role="tooltip">
                <p class="font-mono text-[10px] uppercase tracking-wider text-white/60">{{ c.real === c.pred ? 'Acierto' : 'Error' }}</p>
                <p><span class="text-white/70">real</span> {{ c.real }}</p>
                <p><span class="text-white/70">predicha</span> {{ c.pred }}</p>
                <p class="mt-1 font-mono text-sm font-semibold">{{ c.conteo }} {{ c.conteo === 1 ? 'consulta' : 'consultas' }}</p>
              </div>
            }
          </div>
          <p class="mt-3 min-h-6 text-sm" aria-live="polite">
            @if (celda(); as c) {
              <span class="font-medium">{{ c.real }}</span> → <span class="font-medium">{{ c.pred }}</span>:
              <span class="mono">{{ c.conteo }}</span> {{ c.conteo === 1 ? 'consulta' : 'consultas' }}
              @if (c.real !== c.pred) { <span class="text-tenue">· clic para ver los errores de la clase</span> }
            } @else {
              <span class="text-tenue">Sin celda seleccionada.</span>
            }
          </p>
        </div>

        <div class="panel">
          <h2 class="mb-3 font-semibold">Pares más confundidos</h2>
          <app-estado [cargando]="p.isLoading()" [error]="p.error()" (reintentar)="p.reload()">
            <ol class="stagger-children grid gap-1.5 text-sm">
              @for (x of p.value() ?? []; track x.real + x.pred; let k = $index) {
                <li>
                  <button type="button" class="group w-full rounded-lg px-2 py-1.5 text-left transition hover:bg-acento-suave"
                          [class.bg-acento-suave]="claseErrores() === x.real"
                          (mouseenter)="resaltar(x)" (mouseleave)="resaltado.set(null)" (focus)="resaltar(x)" (blur)="resaltado.set(null)"
                          (click)="verErrores(x.real)">
                    <span class="flex items-baseline gap-2">
                      <span class="mono w-5 shrink-0 text-right text-mal">{{ x.conteo }}</span>
                      <span class="min-w-0">{{ x.real }} <span class="text-tenue">→</span> {{ x.pred }}</span>
                    </span>
                    <span class="ml-7 mt-1 block h-1 overflow-hidden rounded-full bg-mal-suave">
                      <span class="block h-full rounded-full bg-mal transition-[width] duration-700 ease-out" [style.transition-delay.ms]="k * 40"
                            [style.width.%]="listo() ? (x.conteo / maxPar()) * 100 : 0"></span>
                    </span>
                  </button>
                </li>
              }
            </ol>
          </app-estado>
        </div>
      </section>

      @if (claseErrores()) {
        <section class="panel mt-6 animate-fadeInUp">
          <div class="mb-3 flex items-center justify-between gap-3">
            <h2 class="font-semibold">Errores de «{{ claseErrores() }}»</h2>
            <button type="button" class="text-sm text-tenue hover:text-texto" (click)="claseErrores.set(null)">Cerrar</button>
          </div>
          <app-estado [cargando]="e.isLoading()" [error]="e.error()" [vacio]="!e.value()?.items?.length" (reintentar)="e.reload()">
            <div class="overflow-x-auto">
              <table class="tabla min-w-[36rem]">
                <thead><tr><th>Consulta</th><th>Predicha</th><th class="text-right">Confianza</th></tr></thead>
                <tbody class="stagger-children">
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
  protected readonly celda = signal<{ real: string; pred: string; conteo: number; i: number; j: number } | null>(null);
  protected readonly tip = signal({ x: 0, y: 0 });
  protected readonly resaltado = signal<{ i: number; j: number } | null>(null);
  protected readonly listo = signal(false);
  protected readonly pct = pct;

  private readonly lienzo = viewChild<ElementRef<HTMLCanvasElement>>('lienzo');
  private readonly marco = viewChild<ElementRef<HTMLDivElement>>('marco');
  private inicio = performance.now();
  private cuadro = 0;
  private dibujando = false;

  private readonly matriz = computed(() => {
    const d = this.m.value();
    if (!d) return null;
    const n = d.clases.length;
    const x = new Uint16Array(n * n);
    for (const [i, j, c] of d.celdas) x[i * n + j] = c;
    const fuera: Celda[] = d.celdas.filter(([i, j]) => i !== j).map(([i, j, c]) => ({ i, j, c })).sort((a, b) => a.c - b.c);
    const maxFuera = Math.max(1, ...fuera.map((f) => f.c));
    const maxDiag = Math.max(1, ...d.celdas.filter(([i, j]) => i === j).map(([, , c]) => c));
    return { n, x, clases: d.clases, fuera, maxFuera, maxDiag, top: fuera.slice(-3) };
  });
  protected readonly n = computed(() => this.matriz()?.n ?? 0);
  protected readonly aciertos = computed(() => (this.m.value()?.celdas ?? []).filter(([i, j]) => i === j).reduce((s, [, , c]) => s + c, 0));
  protected readonly errores = computed(() => (this.m.value()?.celdas ?? []).filter(([i, j]) => i !== j).reduce((s, [, , c]) => s + c, 0));
  protected readonly celdasError = computed(() => this.matriz()?.fuera.length ?? 0);
  protected readonly maxPar = computed(() => Math.max(1, ...(this.p.value() ?? []).map((x) => x.conteo)));

  constructor() {
    inject(DestroyRef).onDestroy(() => cancelAnimationFrame(this.cuadro));
    afterRenderEffect(() => {
      // Arranca el bucle de dibujo cuando hay datos y lienzo; también al cambiar el resaltado o el cursor
      if (this.matriz() && this.lienzo()) {
        this.celda(); this.resaltado();
        this.arrancar();
      }
      if (this.p.hasValue() && !this.listo()) requestAnimationFrame(() => this.listo.set(true));
    });
  }

  protected repetir() {
    this.inicio = performance.now();
    this.arrancar();
  }

  private arrancar() {
    if (this.dibujando) return;
    this.dibujando = true;
    const paso = (t: number) => {
      if (this.dibujar(t)) this.cuadro = requestAnimationFrame(paso);
      else this.dibujando = false;
    };
    this.cuadro = requestAnimationFrame(paso);
  }

  /** Dibuja un cuadro. Devuelve true mientras haya algo que animar. */
  private dibujar(ahora: number): boolean {
    const mat = this.matriz();
    const canvas = this.lienzo()?.nativeElement;
    if (!mat || !canvas) return false;
    const lado = canvas.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(lado * dpr)) canvas.width = canvas.height = Math.round(lado * dpr);
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, lado, lado);

    const quieto = sinMovimiento();
    const t = quieto ? 1e9 : ahora - this.inicio;
    const s = lado / mat.n;
    const hover = this.celda();
    const foco = this.resaltado() ?? (hover ? { i: hover.i, j: hover.j } : null);

    // Bandas de fila y columna bajo el cursor
    if (foco) {
      ctx.fillStyle = 'rgba(4, 32, 44, 0.06)';
      ctx.fillRect(0, foco.i * s, lado, s);
      ctx.fillRect(foco.j * s, 0, s, lado);
    }
    // Cuadrícula tenue cada 7 clases
    ctx.strokeStyle = 'rgba(4, 32, 44, 0.05)';
    ctx.lineWidth = 1;
    for (let k = 0; k <= mat.n; k += 7) {
      ctx.beginPath(); ctx.moveTo(k * s, 0); ctx.lineTo(k * s, lado); ctx.moveTo(0, k * s); ctx.lineTo(lado, k * s); ctx.stroke();
    }

    const celda = (i: number, j: number, color: string, escala: number) => {
      if (escala <= 0) return;
      const tam = Math.max(1, (s - 0.6) * escala);
      ctx.fillStyle = color;
      ctx.fillRect(j * s + (s - tam) / 2, i * s + (s - tam) / 2, tam, tam);
    };

    // 1) Diagonal: se traza de la esquina superior izquierda a la inferior derecha
    for (let i = 0; i < mat.n; i++) {
      const c = mat.x[i * mat.n + i];
      if (c) celda(i, i, `rgba(4, 32, 44, ${0.3 + 0.7 * (c / mat.maxDiag)})`, suave((t - (i / mat.n) * DURACION_DIAGONAL) / 220));
    }
    // 2) Errores: aparecen del más leve al más grave, con un pequeño rebote
    mat.fuera.forEach((f, k) => {
      const x = Math.min(1, Math.max(0, (t - DURACION_DIAGONAL - (k / mat.fuera.length) * DURACION_ERRORES) / 260));
      const rebote = x < 1 ? 1 + 0.35 * Math.sin(x * Math.PI) : 1;
      celda(f.i, f.j, `rgba(220, 38, 38, ${0.3 + 0.7 * Math.sqrt(f.c / mat.maxFuera)})`, x * rebote);
    });

    const fin = DURACION_DIAGONAL + DURACION_ERRORES + 300;
    // 3) Los tres pares más confundidos laten con un anillo que se expande
    if (t > fin && !quieto) {
      const fase = (ahora / 1400) % 1;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = `rgba(220, 38, 38, ${0.6 * (1 - fase)})`;
      for (const f of mat.top) {
        const r = s * (0.8 + fase * 1.8);
        ctx.strokeRect(f.j * s + s / 2 - r / 2, f.i * s + s / 2 - r / 2, r, r);
      }
    }
    // 4) Celda en foco: ampliada, con sombra y contorno blanco
    if (foco && foco.i >= 0 && foco.j >= 0) {
      const c = mat.x[foco.i * mat.n + foco.j];
      const tam = s * 2.4;
      const x0 = foco.j * s + s / 2 - tam / 2, y0 = foco.i * s + s / 2 - tam / 2;
      ctx.fillStyle = c ? (foco.i === foco.j ? '#04202C' : '#DC2626') : '#DFE4E0';
      ctx.shadowColor = 'rgba(4, 32, 44, 0.35)';
      ctx.shadowBlur = 10;
      ctx.fillRect(x0, y0, tam, tam);
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, tam, tam);
    }
    return !quieto;  // con movimiento: la matriz sigue viva (latido); sin movimiento: un solo cuadro
  }

  protected apuntar(ev: MouseEvent) {
    const mat = this.matriz();
    const canvas = this.lienzo()?.nativeElement;
    if (!mat || !canvas) return;
    const r = canvas.getBoundingClientRect();
    const j = Math.floor(((ev.clientX - r.left) / r.width) * mat.n);
    const i = Math.floor(((ev.clientY - r.top) / r.height) * mat.n);
    if (i < 0 || j < 0 || i >= mat.n || j >= mat.n) return;
    this.celda.set({ real: mat.clases[i], pred: mat.clases[j], conteo: mat.x[i * mat.n + j], i, j });
    const y = ev.clientY - r.top;
    this.tip.set({ x: Math.min(Math.max(ev.clientX - r.left, 90), r.width - 90), y: y > r.height * 0.7 ? y - 110 : y + 18 });
    if (ev.type === 'click' && i !== j) this.verErrores(mat.clases[i]);
  }

  protected salir() {
    this.celda.set(null);
  }

  protected resaltar(x: Par) {
    const mat = this.matriz();
    if (mat) this.resaltado.set({ i: mat.clases.indexOf(x.real), j: mat.clases.indexOf(x.pred) });
  }

  protected verErrores(clase: string) {
    this.claseErrores.set(clase);
  }
}
