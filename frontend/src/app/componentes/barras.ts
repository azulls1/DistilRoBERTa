import { Component, afterNextRender, computed, input, signal } from '@angular/core';

export interface Barra {
  etiqueta: string;
  valor: number;
  texto?: string;
  tono?: 'acento' | 'bien' | 'mal' | 'tenue';
}

/** Barras horizontales accesibles que crecen al aparecer (sin librería de gráficas). */
@Component({
  selector: 'app-barras',
  template: `
    <ol class="grid gap-1.5" [attr.aria-label]="titulo()">
      @for (b of datos(); track b.etiqueta; let i = $index) {
        <li class="group grid grid-cols-[minmax(0,11rem)_1fr_auto] items-center gap-3 text-sm">
          <span class="truncate text-tenue transition group-hover:text-forest" [title]="b.etiqueta">{{ b.etiqueta }}</span>
          <span class="h-2.5 overflow-hidden rounded-full bg-acento-suave">
            <span class="block h-full rounded-full transition-[width] duration-700 ease-out"
                  [class]="color(b.tono)" [style.transition-delay.ms]="i * 25"
                  [style.width.%]="montado() ? (b.valor / maximo()) * 100 : 0"></span>
          </span>
          <span class="mono tabular-nums text-evergreen">{{ b.texto ?? b.valor }}</span>
        </li>
      }
    </ol>
  `,
})
export class Barras {
  readonly datos = input.required<Barra[]>();
  readonly titulo = input('');
  readonly max = input<number | null>(null);
  protected readonly montado = signal(false);
  protected readonly maximo = computed(() => this.max() ?? Math.max(1, ...this.datos().map((d) => d.valor)));

  constructor() {
    afterNextRender(() => requestAnimationFrame(() => this.montado.set(true)));
  }

  protected color(t: Barra['tono']) {
    return { bien: 'bg-bien', mal: 'bg-mal', tenue: 'bg-moss', acento: 'bg-forest' }[t ?? 'acento'];
  }
}
