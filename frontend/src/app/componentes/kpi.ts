import { Component, input } from '@angular/core';

@Component({
  selector: 'app-kpi',
  template: `
    <div class="panel flex flex-col gap-1">
      <span class="text-xs font-medium uppercase tracking-wide text-tenue">{{ etiqueta() }}</span>
      <span class="text-2xl font-semibold tabular-nums">{{ valor() }}</span>
      @if (detalle()) {
        <span class="text-xs text-tenue">{{ detalle() }}</span>
      }
    </div>
  `,
})
export class Kpi {
  readonly etiqueta = input.required<string>();
  readonly valor = input.required<string>();
  readonly detalle = input('');
}
