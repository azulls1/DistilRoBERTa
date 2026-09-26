import { Component, computed, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Encabezado } from '../componentes/encabezado';
import { Narracion } from '../componentes/narracion';
import { NARRACIONES } from '../core/narraciones';

import { Clase } from '../core/modelos';
import { dec } from '../core/formato';

type Columna = 'nombre' | 'precision' | 'recall' | 'f1' | 'errores' | 'n_train';

@Component({
  selector: 'app-clases',
  imports: [Encabezado, Narracion, Estado],
  template: `
    <app-encabezado titulo="Desempeño por clase" icono="lista" etiqueta="C2 · 2.5 pts">
      <span entrada>Precision, recall y F1 de cada una de las 77 intenciones sobre las 40 consultas de prueba de cada clase.
        En verde, las siete mejor clasificadas; en rojo, las siete con más errores.</span>
    </app-encabezado>
    <app-narracion class="mb-6 block animate-fadeInUp" [src]="narracion.src" [titulo]="narracion.titulo" [transcripcion]="narracion.texto" />

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.length" (reintentar)="r.reload()">
      <section class="grid gap-6 lg:grid-cols-2">
        @for (g of grupos(); track g.titulo) {
          <div class="panel">
            <h2 class="mb-3 font-semibold" [class]="g.color">{{ g.titulo }}</h2>
            <ol class="grid gap-2 text-sm">
              @for (c of g.lista; track c.id; let i = $index) {
                <li class="flex items-baseline justify-between gap-3">
                  <span class="truncate"><span class="mono text-tenue">{{ i + 1 }}.</span> {{ c.nombre }}</span>
                  <span class="mono shrink-0 tabular-nums">F1 {{ dec(c.f1) }} · {{ c.errores }} err.</span>
                </li>
              }
            </ol>
          </div>
        }
      </section>

      <section class="panel mt-6">
        <div class="mb-4 flex flex-wrap items-end justify-between gap-3">
          <h2 class="font-semibold">Las 77 clases</h2>
          <label class="grid gap-1 text-sm">
            <span class="text-tenue">Filtrar por nombre</span>
            <input class="campo w-64" type="search" placeholder="p. ej. card, transfer, top up"
                   [value]="filtro()" (input)="filtro.set($any($event.target).value)" />
          </label>
        </div>
        <div class="overflow-x-auto">
          <table class="tabla min-w-[42rem]">
            <thead>
              <tr>
                @for (col of columnas; track col.id) {
                  <th [class.text-right]="col.id !== 'nombre'" [attr.aria-sort]="ariaOrden(col.id)">
                    <button type="button" class="inline-flex items-center gap-1 hover:text-texto" (click)="ordenar(col.id)">
                      {{ col.texto }} <span aria-hidden="true">{{ flecha(col.id) }}</span>
                    </button>
                  </th>
                }
              </tr>
            </thead>
            <tbody class="tabular-nums">
              @for (c of filas(); track c.id) {
                <tr [class.bg-bien-suave]="c.categoria === 'mejor'" [class.bg-mal-suave]="c.categoria === 'peor'">
                  <td class="font-medium">{{ c.nombre }}</td>
                  <td class="text-right">{{ dec(c.precision) }}</td>
                  <td class="text-right">{{ dec(c.recall) }}</td>
                  <td class="text-right">
                    <span class="inline-flex items-center gap-2">
                      <span class="hidden h-1.5 w-16 rounded-full bg-borde sm:inline-block"><span class="block h-full rounded-full bg-acento" [style.width.%]="c.f1 * 100"></span></span>
                      {{ dec(c.f1) }}
                    </span>
                  </td>
                  <td class="text-right">{{ c.errores }}</td>
                  <td class="text-right">{{ c.n_train }}</td>
                </tr>
              } @empty {
                <tr><td colspan="6" class="py-6 text-center text-tenue">Ninguna clase coincide con «{{ filtro() }}».</td></tr>
              }
            </tbody>
          </table>
        </div>
      </section>
    </app-estado>
  `,
})
export class ClasesPagina {
  protected readonly narracion = NARRACIONES.transformer;
  protected readonly r = httpResource<Clase[]>(() => '/api/clases');
  protected readonly dec = dec;
  protected readonly filtro = signal('');
  protected readonly orden = signal<{ col: Columna; asc: boolean }>({ col: 'f1', asc: true });
  protected readonly columnas: { id: Columna; texto: string }[] = [
    { id: 'nombre', texto: 'Clase' }, { id: 'precision', texto: 'Precision' }, { id: 'recall', texto: 'Recall' },
    { id: 'f1', texto: 'F1' }, { id: 'errores', texto: 'Errores' }, { id: 'n_train', texto: 'Ejemplos train' },
  ];

  protected readonly grupos = computed(() => {
    const c = this.r.value() ?? [];
    return [
      { titulo: 'Siete mejor clasificadas', color: 'text-bien',
        lista: c.filter((x) => x.categoria === 'mejor').sort((a, b) => b.f1 - a.f1 || a.errores - b.errores) },
      { titulo: 'Siete con más errores', color: 'text-mal',
        lista: c.filter((x) => x.categoria === 'peor').sort((a, b) => b.errores - a.errores || a.f1 - b.f1) },
    ];
  });

  protected readonly filas = computed(() => {
    const f = this.filtro().trim().toLowerCase().replaceAll(' ', '_');
    const { col, asc } = this.orden();
    return (this.r.value() ?? [])
      .filter((c) => !f || c.nombre.toLowerCase().includes(f))
      .sort((a, b) => {
        const d = col === 'nombre' ? a.nombre.localeCompare(b.nombre) : (a[col] as number) - (b[col] as number);
        return asc ? d : -d;
      });
  });

  protected ordenar(col: Columna) {
    this.orden.update((o) => ({ col, asc: o.col === col ? !o.asc : col === 'nombre' }));
  }
  protected flecha(col: Columna) {
    const o = this.orden();
    return o.col !== col ? '' : o.asc ? '▲' : '▼';
  }
  protected ariaOrden(col: Columna) {
    const o = this.orden();
    return o.col !== col ? 'none' : o.asc ? 'ascending' : 'descending';
  }
}
