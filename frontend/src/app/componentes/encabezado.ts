import { Component, input } from '@angular/core';
import { Icono, NombreIcono } from './icono';

/** Encabezado común de página: etiqueta del criterio, icono, título y entrada. */
@Component({
  selector: 'app-encabezado',
  imports: [Icono],
  template: `
    <header class="mb-8 flex flex-wrap items-start justify-between gap-6 animate-fadeInUp">
      <div class="flex max-w-3xl items-start gap-4">
        <span class="mt-1 hidden h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-forest text-white shadow-[0_8px_24px_-8px_rgba(4,32,44,0.5)] sm:flex">
          <app-icono [nombre]="icono()" clase="h-6 w-6" />
        </span>
        <div class="grid gap-2">
          @if (etiqueta()) {
            <span class="w-fit rounded-full bg-acento-suave px-2.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-[0.12em] text-pine">{{ etiqueta() }}</span>
          }
          <h1 class="titulo sm:text-3xl">{{ titulo() }}</h1>
          <p class="text-[15px] leading-relaxed text-tenue"><ng-content select="[entrada]" /></p>
        </div>
      </div>
      <ng-content />
    </header>
  `,
})
export class Encabezado {
  readonly titulo = input.required<string>();
  readonly icono = input.required<NombreIcono>();
  readonly etiqueta = input('');
}
