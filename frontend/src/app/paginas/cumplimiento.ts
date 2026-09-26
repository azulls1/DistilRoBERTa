import { Component, computed, signal } from '@angular/core';
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
      <span entrada>Cada exigencia del documento <em>SCA_individual.docx</em>, más el nivel 4 («destacado») de la rúbrica
        detallada de Moodle, agrupada por criterio, con la sección del notebook que la cumple, la página de este portal
        donde se ve y la evidencia medida.</span>
    </app-encabezado>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.length" (reintentar)="r.reload()">
      <!-- Resumen por criterio -->
      <section class="stagger-children grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        @for (g of grupos(); track g.criterio) {
          <button type="button" (click)="elegir(g.criterio)" [attr.aria-pressed]="criterio() === g.criterio" [attr.aria-controls]="'detalle-cumplimiento'"
             class="card-stat group relative overflow-hidden text-left transition hover:-translate-y-0.5"
             [class]="criterio() === g.criterio ? '!border-forest ring-2 ring-forest/20' : ''">
            <div class="flex items-start justify-between gap-2">
              <p class="card-stat__label">{{ g.criterio }} · {{ g.nombre }}</p>
              <span class="flex h-7 w-7 items-center justify-center rounded-full bg-bien-suave text-bien"><app-icono nombre="check" clase="h-4 w-4" [grosor]="2.5" /></span>
            </div>
            <p class="card-stat__value font-display">{{ g.cumplidos }} / {{ g.total }}</p>
            <p class="card-stat__desc">{{ g.puntos != null ? g.puntos + ' pts · ' + g.peso + ' % de la nota' : 'Formato de entrega y notas técnicas' }}</p>
            @if (g.peso) {
              <span class="mt-3 block h-1.5 overflow-hidden rounded-full bg-acento-suave"><span class="block h-full rounded-full bg-forest" [style.width.%]="g.peso * 100 / 30"></span></span>
            }
          </button>
        }
      </section>

      <p class="mt-6 flex items-center gap-2 rounded-xl border border-bien/20 bg-bien-suave px-4 py-3 text-sm text-bien">
        <app-icono nombre="checkCirculo" clase="h-5 w-5" /> {{ totalCumplidos() }} de {{ r.value()?.length }} exigencias cumplidas
        ({{ porFuente('Enunciado') }} del enunciado · {{ porFuente('Rúbrica detallada') }} del nivel 4 de la rúbrica · {{ porFuente('Solicitud') }} solicitud), con evidencia en el notebook y en este portal.
      </p>

      <!-- Detalle: un criterio a la vez (se elige en las tarjetas) -->
      @if (grupoActual(); as g) {
        <section id="detalle-cumplimiento" class="mt-8 scroll-mt-24 animate-fadeIn">
          <h2 class="seccion">{{ g.criterio }} · {{ g.nombre }}</h2>
          <ol class="panel !p-0 divide-y divide-fog/60 overflow-hidden">
            @for (q of g.items; track q.orden) {
              <li class="grid gap-2 px-4 py-3 transition hover:bg-acento-suave/50 md:grid-cols-[1.5rem_1fr_16rem] md:items-start">
                <span class="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-forest text-white"><app-icono nombre="check" clase="h-3 w-3" [grosor]="3" /></span>
                <div>
                  <p class="font-medium text-forest">{{ q.requisito }}
                    @if (q.fuente !== 'Enunciado') {
                      <span class="ml-1 rounded px-1.5 py-0.5 align-middle font-mono text-[10px]"
                            [class]="q.fuente === 'Rúbrica detallada' ? 'bg-forest text-white' : 'bg-aviso-suave text-aviso'">{{ q.fuente === 'Rúbrica detallada' ? 'Rúbrica · nivel 4' : 'Solicitud' }}</span>
                    }
                  </p>
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
          <div class="mt-3 flex items-center justify-between text-sm">
            <button type="button" class="inline-flex items-center gap-1 text-pine hover:underline disabled:opacity-30 disabled:no-underline" [disabled]="indice() === 0" (click)="mover(-1)">← {{ grupos()[indice() - 1]?.criterio ?? '' }}</button>
            <span class="font-mono text-[11px] text-moss">criterio {{ indice() + 1 }} de {{ grupos().length }}</span>
            <button type="button" class="inline-flex items-center gap-1 text-pine hover:underline disabled:opacity-30 disabled:no-underline" [disabled]="indice() === grupos().length - 1" (click)="mover(1)">{{ grupos()[indice() + 1]?.criterio ?? '' }} →</button>
          </div>
        </section>
      }
    </app-estado>
  `,
})
export class CumplimientoPagina {
  protected readonly r = httpResource<Requisito[]>(() => '/api/cumplimiento');
  protected porFuente(f: Requisito['fuente']) {
    return (this.r.value() ?? []).filter((q) => q.fuente === f).length;
  }
  protected readonly criterio = signal<string | null>(null);
  protected readonly indice = computed(() => Math.max(0, this.grupos().findIndex((g) => g.criterio === this.criterio())));
  protected readonly grupoActual = computed(() => this.grupos()[this.indice()] ?? null);
  protected elegir(c: string) {
    this.criterio.set(c);
    const el = document.getElementById('detalle-cumplimiento');
    if (el && el.getBoundingClientRect().top > innerHeight * 0.8) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  protected mover(d: number) {
    const g = this.grupos()[this.indice() + d];
    if (g) this.elegir(g.criterio);
  }
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
