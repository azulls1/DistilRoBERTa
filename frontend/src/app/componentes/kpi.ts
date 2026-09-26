import { Component, input } from '@angular/core';

/** Tarjeta de indicador (card-stat del Forest DS). */
@Component({
  selector: 'app-kpi',
  template: `
    <div class="card-stat h-full min-w-0">
      <p class="card-stat__label">{{ etiqueta() }}</p>
      <p class="card-stat__value font-display tabular-nums">{{ valor() }}</p>
      @if (detalle()) {
        <p class="card-stat__desc truncate" [title]="detalle()">{{ detalle() }}</p>
      }
    </div>
  `,
})
export class Kpi {
  readonly etiqueta = input.required<string>();
  readonly valor = input.required<string>();
  readonly detalle = input('');
}
