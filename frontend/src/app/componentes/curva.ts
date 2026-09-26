import { Component, computed, input } from '@angular/core';

export interface Serie {
  nombre: string;
  valores: number[];
  clase: string; // clase de color de Tailwind para stroke/fill
}

/** Gráfica de líneas en SVG para las curvas de entrenamiento (x = época). */
@Component({
  selector: 'app-curva',
  template: `
    <figure class="grid gap-2">
      <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" class="h-auto w-full" role="img" [attr.aria-label]="titulo()">
        @for (t of ticksY(); track t.v) {
          <line [attr.x1]="P" [attr.x2]="W - 8" [attr.y1]="t.y" [attr.y2]="t.y" class="stroke-borde" stroke-dasharray="3 4" />
          <text [attr.x]="P - 6" [attr.y]="t.y + 4" text-anchor="end" class="fill-tenue text-[10px]">{{ t.v.toFixed(2) }}</text>
        }
        @for (x of ticksX(); track x.e) {
          <text [attr.x]="x.x" [attr.y]="H - 6" text-anchor="middle" class="fill-tenue text-[10px]">{{ x.e }}</text>
        }
        @for (s of trazos(); track s.nombre) {
          <polyline [attr.points]="s.puntos" fill="none" stroke-width="2" [class]="s.clase" stroke="currentColor" />
          @for (p of s.xy; track $index) {
            <circle [attr.cx]="p[0]" [attr.cy]="p[1]" r="3" [class]="s.clase" fill="currentColor"><title>{{ s.nombre }}: {{ p[2].toFixed(4) }}</title></circle>
          }
        }
      </svg>
      <figcaption class="flex flex-wrap gap-4 text-xs text-tenue">
        @for (s of series(); track s.nombre) {
          <span class="inline-flex items-center gap-1.5"><span class="h-2 w-4 rounded" [class]="s.clase" style="background: currentColor"></span>{{ s.nombre }}</span>
        }
        <span>eje x: época</span>
      </figcaption>
    </figure>
  `,
})
export class Curva {
  readonly series = input.required<Serie[]>();
  readonly epocas = input.required<number[]>();
  readonly titulo = input('');
  protected readonly W = 520;
  protected readonly H = 220;
  protected readonly P = 40;

  private readonly rango = computed(() => {
    const v = this.series().flatMap((s) => s.valores);
    const min = Math.min(...v), max = Math.max(...v);
    const m = (max - min) * 0.1 || 0.05;
    return [Math.max(0, min - m), max + m];
  });

  private x(i: number) {
    const n = Math.max(1, this.epocas().length - 1);
    return this.P + (i / n) * (this.W - this.P - 16);
  }
  private y(v: number) {
    const [a, b] = this.rango();
    return 10 + (1 - (v - a) / (b - a)) * (this.H - 36);
  }

  protected readonly trazos = computed(() =>
    this.series().map((s) => {
      const xy = s.valores.map((v, i) => [this.x(i), this.y(v), v] as [number, number, number]);
      return { ...s, xy, puntos: xy.map((p) => `${p[0]},${p[1]}`).join(' ') };
    }),
  );
  protected readonly ticksY = computed(() => {
    const [a, b] = this.rango();
    return [0, 0.25, 0.5, 0.75, 1].map((f) => ({ v: a + f * (b - a), y: this.y(a + f * (b - a)) }));
  });
  protected readonly ticksX = computed(() => this.epocas().map((e, i) => ({ e, x: this.x(i) })));
}
