import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { HttpClient, httpResource } from '@angular/common/http';
import { Encabezado } from '../componentes/encabezado';
import { Estado } from '../componentes/estado';
import { Icono, NombreIcono } from '../componentes/icono';
import { Kpi } from '../componentes/kpi';
import { ArchivoEntregable, Entregables, Paquete } from '../core/modelos';

const ICONO_TIPO: Record<string, NombreIcono> = {
  Notebook: 'libro', PDF: 'archivo', Figura: 'imagen', Datos: 'datos', 'Código': 'codigo', Documento: 'archivo', 'Especificación': 'escudo',
};
const NOMBRE_CRITERIO: Record<string, string> = {
  Entrega: 'Entrega', C1: 'C1 · EDA', C2: 'C2 · Transformer', C3: 'C3 · Prompt', C5: 'C5 · Código y referencias',
};

@Component({
  selector: 'app-entregables',
  imports: [Encabezado, Estado, Icono, Kpi],
  template: `
    <app-encabezado titulo="Entregables" icono="paquete" etiqueta="Entrega · Moodle">
      <span entrada>Todo lo que compone la entrega de la Actividad 2: el notebook y su PDF (lo que pide el enunciado),
        las figuras extraídas del notebook ejecutado, los datos de resultados, el código y las especificaciones.
        Cada archivo lleva su criterio de la rúbrica y su huella SHA-256.</span>
    </app-encabezado>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.archivos?.length" (reintentar)="r.reload()">
      @if (r.value(); as d) {
        <section class="stagger-children grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Resumen de entregables">
          <app-kpi etiqueta="Archivos" [valor]="'' + d.archivos.length" detalle="en el catálogo" icono="archivo" />
          <app-kpi etiqueta="Tamaño total" [valor]="mb(d.total_bytes)" detalle="sin comprimir" icono="datos" />
          <app-kpi etiqueta="Figuras" [valor]="'' + cuenta('Figura')" detalle="extraídas del notebook" icono="imagen" />
          <app-kpi etiqueta="Paquetes generados" [valor]="'' + d.recientes.length" detalle="últimos registrados" icono="paquete" />
        </section>

        <!-- Lo que se sube a Moodle -->
        <h2 class="seccion mt-8">Lo que se sube a Moodle</h2>
        <section class="grid gap-4 md:grid-cols-2">
          @for (a of principales(); track a.ruta) {
            <a [href]="url(a)" download class="panel group flex items-center gap-4 transition hover:-translate-y-0.5">
              <span class="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-forest text-white transition group-hover:scale-105">
                <app-icono [nombre]="icono(a.tipo)" clase="h-7 w-7" />
              </span>
              <span class="min-w-0 flex-1">
                <span class="block font-display font-semibold text-forest">{{ a.nombre }}</span>
                <span class="block truncate font-mono text-[11px] text-moss">{{ a.ruta }} · {{ kb(a.bytes) }}</span>
                <span class="mt-1 block text-xs text-tenue">{{ a.detalle }}</span>
              </span>
              <app-icono nombre="descarga" clase="h-5 w-5 text-pine transition group-hover:translate-y-0.5" />
            </a>
          }
        </section>

        <!-- Paquete ZIP -->
        <h2 class="seccion mt-8">Paquete completo</h2>
        <section class="card-hero !p-6 !text-left sm:!p-8">
          <div class="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-white/5 blur-2xl orbe"></div>
          <div class="relative grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
            <div>
              <p class="font-display text-xl font-semibold text-white">Generar el ZIP de la entrega</p>
              <p class="mt-1 max-w-xl text-sm text-white/70">Un worker de Celery empaqueta los {{ d.archivos.length }} archivos del catálogo más un
                manifiesto con el SHA-256 de cada uno, y calcula la huella del ZIP. Tarda unos segundos.</p>
              @if (paquete(); as p) {
                <div class="mt-4 rounded-lg border border-white/15 bg-white/5 p-3 font-mono text-[11px] text-white/80">
                  @if (p.estado === 'completada') {
                    <p class="text-emerald-300">✓ {{ p.archivo }}</p>
                    <p>{{ p.n_archivos }} archivos · {{ mb(p.bytes ?? 0) }} comprimido</p>
                    <p class="break-all">SHA-256 {{ p.sha256 }}</p>
                  } @else if (p.estado === 'error') {
                    <p class="text-red-300">✗ {{ p.error }}</p>
                  } @else {
                    <p class="flex items-center gap-2"><span class="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white"></span> Empaquetando…</p>
                  }
                </div>
              }
            </div>
            <div class="flex flex-col gap-2">
              <button type="button" (click)="generar()" [disabled]="generando()"
                      class="inline-flex items-center justify-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-forest transition hover:-translate-y-px disabled:opacity-60">
                <app-icono nombre="paquete" clase="h-4 w-4" /> {{ generando() ? 'Generando…' : 'Generar paquete' }}</button>
              @if (paquete()?.estado === 'completada') {
                <a [href]="'/api/entregables/paquete/' + paquete()!.id + '/descargar'" download
                   class="inline-flex items-center justify-center gap-2 rounded-lg border border-white/25 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-white/10">
                  <app-icono nombre="descarga" clase="h-4 w-4" /> Descargar ZIP</a>
              }
            </div>
          </div>
        </section>

        <!-- Figuras -->
        <h2 class="seccion mt-8">Figuras del notebook</h2>
        <section class="stagger-children grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          @for (f of figuras(); track f.ruta) {
            <a [href]="url(f)" target="_blank" rel="noopener" class="panel group !p-2 transition hover:-translate-y-0.5">
              <span class="block aspect-[4/3] overflow-hidden rounded-lg bg-acento-suave">
                <img [src]="url(f)" [alt]="f.nombre" loading="lazy" class="h-full w-full object-cover object-top transition duration-500 group-hover:scale-105" />
              </span>
              <span class="mt-2 block px-1 text-xs font-medium text-forest">{{ f.nombre }}</span>
              <span class="block px-1 pb-1 font-mono text-[10px] text-moss">{{ f.criterio }} · {{ f.detalle }}</span>
            </a>
          }
        </section>

        <!-- Catálogo -->
        <h2 class="seccion mt-8">Catálogo completo</h2>
        <div class="mb-3 flex flex-wrap gap-2" role="group" aria-label="Filtrar por criterio">
          @for (c of criterios(); track c.id) {
            <button type="button" (click)="filtro.set(c.id)" [attr.aria-pressed]="filtro() === c.id"
                    class="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition"
                    [class]="filtro() === c.id ? 'border-forest bg-forest text-white' : 'border-fog bg-white text-evergreen hover:border-forest'">
              {{ c.texto }} <span class="rounded-full px-1.5 font-mono text-[10px]" [class]="filtro() === c.id ? 'bg-white/20' : 'bg-acento-suave'">{{ c.n }}</span>
            </button>
          }
        </div>
        <section class="panel !p-0 overflow-hidden">
          <ul class="divide-y divide-fog/60">
            @for (a of filtrados(); track a.ruta) {
              <li class="group flex flex-wrap items-center gap-3 px-4 py-3 transition hover:bg-acento-suave/60">
                <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-acento-suave text-pine group-hover:bg-forest group-hover:text-white">
                  <app-icono [nombre]="icono(a.tipo)" clase="h-4 w-4" /></span>
                <span class="min-w-0 flex-1">
                  <span class="flex flex-wrap items-baseline gap-2"><span class="font-medium text-forest">{{ a.nombre }}</span>
                    <span class="rounded bg-acento-suave px-1.5 font-mono text-[10px] text-pine">{{ a.criterio }}</span>
                    <span class="font-mono text-[10px] text-moss">{{ a.tipo }}</span></span>
                  <span class="block text-xs text-tenue">{{ a.detalle }}</span>
                  <span class="block truncate font-mono text-[10px] text-moss" [title]="a.sha256">{{ a.ruta }} · {{ kb(a.bytes) }} · sha256 {{ a.sha256.slice(0, 16) }}…</span>
                </span>
                <a [href]="url(a)" download class="inline-flex items-center gap-1 rounded-lg border border-fog px-3 py-1.5 text-xs text-forest transition hover:border-forest">
                  <app-icono nombre="descarga" clase="h-3.5 w-3.5" /> Descargar</a>
              </li>
            }
          </ul>
        </section>
      }
    </app-estado>
  `,
})
export class EntregablesPagina {
  private readonly http = inject(HttpClient);
  private temporizador: ReturnType<typeof setTimeout> | undefined;
  protected readonly r = httpResource<Entregables>(() => '/api/entregables');
  protected readonly filtro = signal('todos');
  protected readonly generando = signal(false);
  protected readonly generado = signal<Paquete | null>(null);
  protected readonly paquete = computed(() => this.generado() ?? this.r.value()?.ultimo_paquete ?? null);

  protected readonly principales = computed(() => (this.r.value()?.archivos ?? []).filter((a) => a.criterio === 'Entrega'));
  protected readonly figuras = computed(() => (this.r.value()?.archivos ?? []).filter((a) => a.tipo === 'Figura'));
  protected readonly criterios = computed(() => {
    const a = this.r.value()?.archivos ?? [];
    const ids = [...new Set(a.map((x) => x.criterio))];
    return [{ id: 'todos', texto: 'Todos', n: a.length },
      ...ids.map((id) => ({ id, texto: NOMBRE_CRITERIO[id] ?? id, n: a.filter((x) => x.criterio === id).length }))];
  });
  protected readonly filtrados = computed(() => {
    const f = this.filtro();
    return (this.r.value()?.archivos ?? []).filter((a) => f === 'todos' || a.criterio === f);
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => clearTimeout(this.temporizador));
  }

  protected generar() {
    this.generando.set(true);
    this.http.post<{ paquete_id: string }>('/api/entregables/generar', {}).subscribe({
      next: ({ paquete_id }) => this.sondear(paquete_id, Date.now()),
      error: () => { this.generando.set(false); this.generado.set({ estado: 'error', error: 'No se pudo encolar la generación.' } as Paquete); },
    });
  }

  private sondear(id: string, inicio: number) {
    this.http.get<Paquete>(`/api/entregables/paquete/${id}`).subscribe({
      next: (p) => {
        this.generado.set(p);
        if (p.estado === 'pendiente' && Date.now() - inicio < 60000) {
          this.temporizador = setTimeout(() => this.sondear(id, inicio), 700);
        } else {
          this.generando.set(false);
        }
      },
      error: () => this.generando.set(false),
    });
  }

  protected cuenta(tipo: string) {
    return (this.r.value()?.archivos ?? []).filter((a) => a.tipo === tipo).length;
  }
  protected url(a: ArchivoEntregable) {
    return `/api/entregables/archivo?ruta=${encodeURIComponent(a.ruta)}`;
  }
  protected icono(tipo: string): NombreIcono {
    return ICONO_TIPO[tipo] ?? 'archivo';
  }
  protected kb(b: number) {
    return b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1024).toFixed(1)} KB`;
  }
  protected mb(b: number) {
    return `${(b / 1e6).toFixed(1)} MB`;
  }
}
