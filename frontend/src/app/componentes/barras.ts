import { Component, computed, input } from '@angular/core';

export interface Barra {
  etiqueta: string;
  valor: number;
  texto?: string;
  tono?: 'acento' | 'bien' | 'mal' | 'tenue';
}

/** Barras horizontales accesibles (lista con valores legibles, sin librería de gráficas). */
@Component({
  selector: 'app-barras',
  template: `
    <ol class="grid gap-1.5" [attr.aria-label]="titulo()">
      @for (b of datos(); track b.etiqueta) {
        <li class="grid grid-cols-[minmax(0,11rem)_1fr_auto] items-center gap-3 text-sm">
          <span class="truncate text-tenue" [title]="b.etiqueta">{{ b.etiqueta }}</span>
          <span class="h-2.5 rounded-full bg-borde/60">
            <span class="block h-full rounded-full transition-[width] duration-500"
                  [class]="color(b.tono)" [style.width.%]="(b.valor / maximo()) * 100"></span>
          </span>
          <span class="mono tabular-nums">{{ b.texto ?? b.valor }}</span>
        </li>
      }
    </ol>
  `,
})
export class Barras {
  readonly datos = input.required<Barra[]>();
  readonly titulo = input('');
  readonly max = input<number | null>(null);
  protected readonly maximo = computed(() => this.max() ?? Math.max(1, ...this.datos().map((d) => d.valor)));

  protected color(t: Barra['tono']) {
    return { bien: 'bg-bien', mal: 'bg-mal', tenue: 'bg-tenue', acento: 'bg-acento' }[t ?? 'acento'];
  }
}
