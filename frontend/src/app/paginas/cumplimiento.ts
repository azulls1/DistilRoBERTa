import { Component, computed } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { RouterLink } from '@angular/router';
import { Encabezado } from '../componentes/encabezado';
import { Estado } from '../componentes/estado';
import { Icono } from '../componentes/icono';
import { Requisito } from '../core/modelos';

@Component({
  selector: 'app-cumplimiento',
  imports: [Encabezado, Estado, Icono, RouterLink],
  template: `
    <app-encabezado titulo="Cumplimiento del enunciado" icono="escudo" etiqueta="Rúbrica · 10 puntos">
      <span entrada>Cada exigencia del documento <em>SCA_individual.docx</em>, agrupada por criterio de la rúbrica, con la
        sección del notebook que la cumple, la página de este portal donde se ve y la evidencia medida.</span>
    </app-encabezado>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.length" (reintentar)="r.reload()">
      <!-- Resumen por criterio -->
      <section class="stagger-children grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        @for (g of grupos(); track g.criterio) {
          <a [href]="'#' + g.criterio" class="card-stat group relative overflow-hidden transition hover:-translate-y-0.5">
            <div class="flex items-start justify-between gap-2">
              <p class="card-stat__label">{{ g.criterio }} · {{ g.nombre }}</p>
              <span class="flex h-7 w-7 items-center justify-center rounded-full bg-bien-suave text-bien"><app-icono nombre="check" clase="h-4 w-4" [grosor]="2.5" /></span>
            </div>
            <p class="card-stat__value font-display">{{ g.cumplidos }} / {{ g.total }}</p>
            <p class="card-stat__desc">{{ g.puntos != null ? g.puntos + ' pts · ' + g.peso + ' % de la nota' : 'Formato de entrega y notas técnicas' }}</p>
            @if (g.peso) {
              <span class="mt-3 block h-1.5 overflow-hidden rounded-full bg-acento-suave"><span class="block h-full rounded-full bg-forest" [style.width.%]="g.peso * 100 / 30"></span></span>
            }
          </a>
        }
      </section>

      <p class="mt-6 flex items-center gap-2 rounded-xl border border-bien/20 bg-bien-suave px-4 py-3 text-sm text-bien">
        <app-icono nombre="checkCirculo" clase="h-5 w-5" /> {{ totalCumplidos() }} de {{ r.value()?.length }} exigencias cumplidas, con evidencia en el notebook y en este portal.
      </p>

      <!-- Detalle -->
      @for (g of grupos(); track g.criterio) {
        <section class="mt-8 scroll-mt-24" [id]="g.criterio">
          <h2 class="seccion">{{ g.criterio }} · {{ g.nombre }}</h2>
          <ol class="panel !p-0 divide-y divide-fog/60 overflow-hidden">
            @for (q of g.items; track q.orden) {
              <li class="grid gap-2 px-4 py-3 transition hover:bg-acento-suave/50 md:grid-cols-[1.5rem_1fr_16rem] md:items-start">
                <span class="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-forest text-white"><app-icono nombre="check" clase="h-3 w-3" [grosor]="3" /></span>
                <div>
                  <p class="font-medium text-forest">{{ q.requisito }}</p>
                  <p class="mt-0.5 text-sm text-tenue">{{ q.evidencia }}</p>
                </div>
                <div class="flex flex-wrap items-center gap-2 md:justify-end">
                  <span class="rounded bg-acento-suave px-2 py-0.5 font-mono text-[10px] text-pine">Notebook § {{ q.seccion_notebook }}</span>
                  <a [routerLink]="q.ruta_web" class="inline-flex items-center gap-1 rounded border border-fog px-2 py-0.5 font-mono text-[10px] text-forest transition hover:border-forest">
                    {{ q.ruta_web }} <app-icono nombre="flecha" clase="h-3 w-3" /></a>
                </div>
              </li>
            }
          </ol>
        </section>
      }
    </app-estado>
  `,
})
export class CumplimientoPagina {
  protected readonly r = httpResource<Requisito[]>(() => '/api/cumplimiento');
  protected readonly totalCumplidos = computed(() => (this.r.value() ?? []).filter((q) => q.cumplido).length);
  protected readonly grupos = computed(() => {
    const orden: string[] = [];
    const mapa = new Map<string, Requisito[]>();
    for (const q of this.r.value() ?? []) {
      if (!mapa.has(q.criterio)) { mapa.set(q.criterio, []); orden.push(q.criterio); }
      mapa.get(q.criterio)!.push(q);
    }
    return orden.map((c) => {
      const items = mapa.get(c)!;
      return { criterio: c, nombre: items[0].criterio_nombre, puntos: items[0].puntos, peso: items[0].peso,
               total: items.length, cumplidos: items.filter((q) => q.cumplido).length, items };
    });
  });
}
