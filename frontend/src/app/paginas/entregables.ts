import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { HttpClient, httpResource } from '@angular/common/http';
import { Encabezado } from '../componentes/encabezado';
import { Estado } from '../componentes/estado';
import { Icono, NombreIcono } from '../componentes/icono';
import { Kpi } from '../componentes/kpi';
import { Paginador, pagina } from '../componentes/paginador';
import { ArchivoEntregable, Entregables, Paquete } from '../core/modelos';

const ICONO_TIPO: Record<string, NombreIcono> = {
  Notebook: 'libro', PDF: 'archivo', Paquete: 'paquete', Figura: 'imagen', Datos: 'datos', 'Código': 'codigo', Documento: 'archivo', 'Especificación': 'escudo',
};
const NOMBRE_CRITERIO: Record<string, string> = {
  Entrega: 'Entrega', C1: 'C1 · EDA', C2: 'C2 · Transformer', C3: 'C3 · Prompt', C5: 'C5 · Código y referencias',
};

@Component({
  selector: 'app-entregables',
  imports: [Encabezado, Estado, Icono, Kpi, Paginador],
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

        <!-- Entrega para el profesor -->
        @if (d.entrega; as z) {
          <h2 class="seccion mt-8">Entrega para el profesor</h2>
          <section class="card-hero !p-6 !text-left sm:!p-8 animate-fadeInUp">
            <div class="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
              <div class="orbe absolute -right-20 -top-20 h-64 w-64 rounded-full bg-[#5B7065]/25 blur-3xl"></div>
              <div class="orbe orbe--lento absolute -bottom-24 left-10 h-56 w-56 rounded-full bg-[#9EADA3]/15 blur-3xl"></div>
            </div>
            <div class="relative grid gap-6 lg:grid-cols-[1fr_22rem]">
              <div>
                <span class="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] text-white/85">
                  <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400"></span> Listo para subir a Moodle</span>
                <p class="mt-4 font-display text-2xl font-semibold text-white">{{ z.ruta }}</p>
                <p class="mt-2 max-w-xl text-sm text-white/70">El paquete que se entrega: notebook ejecutado, su PDF, las figuras y los resultados,
                  con un LEEME que mapea cada criterio de la rúbrica a su evidencia. Se regenera en cada despliegue a partir de los archivos reales.</p>
                <div class="mt-5 flex flex-wrap gap-3">
                  <a [href]="url(z)" download class="inline-flex items-center gap-2 rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-forest shadow-lg transition hover:-translate-y-0.5">
                    <app-icono nombre="descarga" clase="h-4 w-4" [grosor]="2" /> Descargar entregable ({{ kb(z.bytes) }})</a>
                  <button type="button" (click)="verLeeme.set(!verLeeme())" [attr.aria-expanded]="verLeeme()"
                          class="inline-flex items-center gap-2 rounded-lg border border-white/25 px-4 py-2.5 text-sm text-white transition hover:bg-white/10">
                    <app-icono nombre="libro" clase="h-4 w-4" /> {{ verLeeme() ? 'Ocultar' : 'Ver' }} LEEME</button>
                </div>
                <p class="mt-4 break-all font-mono text-[10px] text-white/50">SHA-256 {{ z.sha256 }}</p>
              </div>
              <div class="rounded-xl border border-white/10 bg-white/5 p-4">
                <p class="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-white/50">Contenido · {{ z.contenido.length }} archivos</p>
                <ul class="grid gap-1 font-mono text-[11px] text-white/85">
                  @for (g of arbol(); track g.carpeta; let k = $index) {
                    <li class="animate-fadeInUp" [style.animation-delay.ms]="200 + k * 90">
                      <span class="flex items-center gap-1.5"><app-icono [nombre]="g.carpeta ? 'paquete' : 'archivo'" clase="h-3.5 w-3.5 text-emerald-300" />
                        {{ g.carpeta || g.archivos[0].nombre }}@if (g.carpeta) {/ <span class="text-white/45">· {{ g.archivos.length }}</span>}</span>
                      @if (g.carpeta) {
                        <ul class="ml-5 mt-0.5 grid gap-0.5 border-l border-white/10 pl-2 text-white/60">
                          @for (a of g.archivos.slice(0, 3); track a.nombre) { <li class="truncate">{{ a.nombre.split('/').pop() }}</li> }
                          @if (g.archivos.length > 3) { <li class="text-white/40">… {{ g.archivos.length - 3 }} más</li> }
                        </ul>
                      }
                    </li>
                  }
                </ul>
              </div>
            </div>
            @if (verLeeme()) {
              <pre class="relative mt-6 max-h-96 overflow-auto whitespace-pre-wrap rounded-xl bg-black/25 p-4 font-mono text-[11px] leading-relaxed text-white/85 animate-fadeIn">{{ z.leeme }}</pre>
            }
          </section>
        }

        <!-- Lo que se sube a Moodle -->
        <h2 class="seccion mt-8">O descarga por separado</h2>
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
        <h2 class="seccion mt-8">Paquete completo del proyecto</h2>
        <section class="card-hero !p-6 !text-left sm:!p-8">
          <div class="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-white/5 blur-2xl orbe"></div>
          <div class="relative grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
            <div>
              <p class="font-display text-xl font-semibold text-white">Generar el ZIP con todo el proyecto</p>
              <p class="mt-1 max-w-xl text-sm text-white/70">Un worker de Celery empaqueta los {{ d.archivos.length }} archivos del catálogo más un
                manifiesto con el SHA-256 de cada uno, y calcula la huella del ZIP. Tarda unos segundos.</p>
              @if (paquete(); as p) {
                <div class="mt-4 rounded-lg border border-white/15 bg-white/5 p-3 font-mono text-[11px] text-white/80">
                  @if (p.estado === 'completada') {
                    <p class="text-emerald-300">✓ {{ p.archivo }}</p>
                    @if (reutilizado()) { <p class="text-white/60">Ya estaba generado con los archivos vigentes; se reutiliza en lugar de crear uno idéntico.</p> }
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
        <h2 id="figuras" class="seccion mt-8 scroll-mt-24">Figuras del notebook</h2>
        <section class="stagger-children grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          @for (f of paginaFiguras(); track f.ruta) {
            <a [href]="url(f)" target="_blank" rel="noopener" class="panel group !p-2 transition hover:-translate-y-0.5">
              <span class="block aspect-[4/3] overflow-hidden rounded-lg bg-acento-suave">
                <img [src]="url(f)" [alt]="f.nombre" loading="lazy" class="h-full w-full object-cover object-top transition duration-500 group-hover:scale-105" />
              </span>
              <span class="mt-2 block px-1 text-xs font-medium text-forest">{{ f.nombre }}</span>
              <span class="block px-1 pb-1 font-mono text-[10px] text-moss">{{ f.criterio }} · {{ f.detalle }}</span>
            </a>
          }
        </section>
        <app-paginador [total]="figuras().length" [(pagina)]="pagFig" [(tamano)]="tamFig" [opciones]="[4, 8]" etiqueta="figuras" ancla="figuras" />

        <!-- Catálogo -->
        <h2 id="catalogo" class="seccion mt-8 scroll-mt-24">Catálogo completo</h2>
        <div class="mb-3 flex flex-wrap gap-2" role="group" aria-label="Filtrar por criterio">
          @for (c of criterios(); track c.id) {
            <button type="button" (click)="filtro.set(c.id); pagCat.set(1)" [attr.aria-pressed]="filtro() === c.id"
                    class="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition"
                    [class]="filtro() === c.id ? 'border-forest bg-forest text-white' : 'border-fog bg-white text-evergreen hover:border-forest'">
              {{ c.texto }} <span class="rounded-full px-1.5 font-mono text-[10px]" [class]="filtro() === c.id ? 'bg-white/20' : 'bg-acento-suave'">{{ c.n }}</span>
            </button>
          }
        </div>
        <section class="panel !p-0 overflow-hidden">
          <ul class="divide-y divide-fog/60">
            @for (a of paginaCatalogo(); track a.ruta) {
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
        <app-paginador [total]="filtrados().length" [(pagina)]="pagCat" [(tamano)]="tamCat" [opciones]="[8, 16, 40]" etiqueta="archivos" ancla="catalogo" />
      }
    </app-estado>
  `,
})
export class EntregablesPagina {
  private readonly http = inject(HttpClient);
  private temporizador: ReturnType<typeof setTimeout> | undefined;
  protected readonly r = httpResource<Entregables>(() => '/api/entregables');
  protected readonly filtro = signal('todos');
  protected readonly pagFig = signal(1);
  protected readonly tamFig = signal(4);
  protected readonly pagCat = signal(1);
  protected readonly tamCat = signal(8);
  protected readonly paginaFiguras = computed(() => pagina(this.figuras(), this.pagFig(), this.tamFig()));
  protected readonly paginaCatalogo = computed(() => pagina(this.filtrados(), this.pagCat(), this.tamCat()));
  protected readonly generando = signal(false);
  /** El servidor devolvió el ZIP vigente en vez de construir uno idéntico. */
  protected readonly reutilizado = signal(false);
  protected readonly generado = signal<Paquete | null>(null);
  protected readonly paquete = computed(() => this.generado() ?? this.r.value()?.ultimo_paquete ?? null);

  protected readonly principales = computed(() => (this.r.value()?.archivos ?? []).filter((a) => a.criterio === 'Entrega' && a.tipo !== 'Paquete'));
  protected readonly verLeeme = signal(false);
  /** Agrupa el contenido real del ZIP por su carpeta numerada. */
  protected readonly arbol = computed(() => {
    const grupos = new Map<string, { nombre: string; bytes: number }[]>();
    for (const a of this.r.value()?.entrega?.contenido ?? []) {
      const carpeta = a.nombre.includes('/') ? a.nombre.split('/')[0] : '';
      const clave = carpeta || a.nombre;
      if (!grupos.has(clave)) grupos.set(clave, []);
      grupos.get(clave)!.push(a);
    }
    return [...grupos.entries()].map(([k, archivos]) => ({ carpeta: archivos[0].nombre.includes('/') ? k : '', archivos }));
  });
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
    this.reutilizado.set(false);
    this.http.post<{ paquete_id: string; reutilizado?: boolean }>('/api/entregables/generar', {}).subscribe({
      next: ({ paquete_id, reutilizado }) => { this.reutilizado.set(!!reutilizado); this.sondear(paquete_id, Date.now()); },
      error: (e) => {
        this.generando.set(false);
        const msg = e?.status === 429 ? 'Demasiadas solicitudes seguidas: espera un minuto e inténtalo de nuevo.' : 'No se pudo encolar la generación.';
        this.generado.set({ estado: 'error', error: msg } as Paquete);
      },
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
