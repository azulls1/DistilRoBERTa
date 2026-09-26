import { Component, DestroyRef, effect, inject, input, signal } from '@angular/core';
import { Icono, NombreIcono } from './icono';

const sinMovimiento = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Tarjeta de indicador (card-stat del Forest DS). La parte numérica del valor cuenta hacia arriba al aparecer. */
@Component({
  selector: 'app-kpi',
  imports: [Icono],
  template: `
    <div class="card-stat group relative h-full min-w-0 overflow-hidden transition duration-300 hover:-translate-y-0.5">
      <div class="flex items-start justify-between gap-2">
        <p class="card-stat__label">{{ etiqueta() }}</p>
        @if (icono()) {
          <span class="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-acento-suave text-pine transition group-hover:bg-forest group-hover:text-white">
            <app-icono [nombre]="icono()!" clase="h-4 w-4" />
          </span>
        }
      </div>
      <p class="card-stat__value font-display tabular-nums" [attr.aria-label]="valor()">{{ mostrado() }}</p>
      @if (detalle()) {
        <p class="card-stat__desc truncate" [title]="detalle()">{{ detalle() }}</p>
      }
      <span class="absolute inset-x-0 bottom-0 h-0.5 origin-left scale-x-0 bg-forest transition-transform duration-500 group-hover:scale-x-100"></span>
    </div>
  `,
})
export class Kpi {
  readonly etiqueta = input.required<string>();
  readonly valor = input.required<string>();
  readonly detalle = input('');
  readonly icono = input<NombreIcono | null>(null);
  protected readonly mostrado = signal('');
  private cuadro = 0;

  constructor() {
    inject(DestroyRef).onDestroy(() => cancelAnimationFrame(this.cuadro));
    effect(() => this.animar(this.valor()));
  }

  /** Anima el primer número del texto conservando prefijo, sufijo, decimales y separador de miles. */
  private animar(texto: string) {
    const m = texto.match(/-?\d[\d,]*(\.\d+)?/);
    if (!m || sinMovimiento()) {
      this.mostrado.set(texto);
      return;
    }
    const objetivo = Number(m[0].replaceAll(',', ''));
    const decimales = m[1] ? m[1].length - 1 : 0;
    const miles = m[0].includes(',');
    const [antes, despues] = [texto.slice(0, m.index), texto.slice(m.index! + m[0].length)];
    const inicio = performance.now();
    const paso = (t: number) => {
      const p = Math.min(1, (t - inicio) / 900);
      const v = objetivo * (1 - Math.pow(1 - p, 3));
      const n = miles ? v.toLocaleString('en-US', { minimumFractionDigits: decimales, maximumFractionDigits: decimales })
                      : v.toFixed(decimales);
      this.mostrado.set(antes + n + despues);
      if (p < 1) this.cuadro = requestAnimationFrame(paso);
    };
    cancelAnimationFrame(this.cuadro);
    this.cuadro = requestAnimationFrame(paso);
  }
}
