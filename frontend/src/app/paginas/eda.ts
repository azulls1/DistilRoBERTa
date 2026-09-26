import { Component, computed, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Encabezado } from '../componentes/encabezado';

import { Barras } from '../componentes/barras';
import { Kpi } from '../componentes/kpi';
import { Paginador, pagina } from '../componentes/paginador';
import { Clase, Eda } from '../core/modelos';
import { dec, num } from '../core/formato';

const NOMBRE_METRICA: Record<string, string> = {
  n_caracteres: 'Caracteres',
  n_palabras: 'Palabras',
  n_tokens: 'Tokens (DistilRoBERTa)',
};

@Component({
  selector: 'app-eda',
  imports: [Encabezado, Estado, Barras, Kpi, Paginador],
  template: `
    <app-encabezado titulo="Análisis exploratorio" icono="grafica" etiqueta="C1 · 1.5 pts">
      <span entrada>{{ num(total()) }} consultas en inglés ({{ num(n('train')) }} de entrenamiento y {{ num(n('test')) }} de prueba). Las frecuencias se calculan sobre el
        texto limpio: minúsculas, sin caracteres especiales y sin stopwords.</span>
    </app-encabezado>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.hasValue()" (reintentar)="r.reload()">
      @if (r.value(); as d) {
        <section class="panel overflow-x-auto">
          <h2 class="mb-3 font-semibold">Longitud de las consultas</h2>
          <table class="tabla min-w-[40rem]">
            <thead>
              <tr><th>Métrica</th><th>Partición</th><th class="text-right">Media</th><th class="text-right">Desv.</th>
                <th class="text-right">Mín.</th><th class="text-right">Q1</th><th class="text-right">Mediana</th>
                <th class="text-right">Q3</th><th class="text-right">Máx.</th></tr>
            </thead>
            <tbody class="tabular-nums">
              @for (e of d.estadisticas; track e.metrica + e.particion) {
                <tr>
                  <td>{{ nombreMetrica(e.metrica) }}</td><td>{{ e.particion }}</td>
                  <td class="text-right">{{ dec(e.mean, 2) }}</td><td class="text-right">{{ dec(e.std, 2) }}</td>
                  <td class="text-right">{{ e.min }}</td><td class="text-right">{{ e.q1 }}</td>
                  <td class="text-right">{{ e.mediana }}</td><td class="text-right">{{ e.q3 }}</td><td class="text-right">{{ e.max }}</td>
                </tr>
              }
            </tbody>
          </table>
        </section>

        <section class="panel mt-6">
          <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 class="font-semibold">N-gramas más frecuentes</h2>
            <div class="flex gap-1 rounded-lg bg-acento-suave p-1" role="tablist" aria-label="Tipo de n-grama">
              @for (t of tiposNgrama(d); track t.id) {
                <button type="button" role="tab" [attr.aria-selected]="ngrama() === t.id" (click)="ngrama.set(t.id)"
                        class="rounded-md px-3 py-1.5 text-xs font-medium transition"
                        [class]="ngrama() === t.id ? 'bg-forest text-white shadow-sm' : 'text-pine hover:bg-white'">{{ t.texto }}</button>
              }
            </div>
          </div>
          @switch (ngrama()) {
            @case ('palabra') { <app-barras titulo="Palabras más frecuentes" [datos]="barras(d.ngramas.palabra.slice(0, 15))" /> }
            @case ('bigrama') { <app-barras titulo="Bigramas más frecuentes" [datos]="barras(d.ngramas.bigrama, 'bien')" /> }
            @case ('trigrama') { <app-barras titulo="Trigramas más frecuentes" [datos]="barras(d.ngramas.trigrama, 'mal')" /> }
          }
        </section>

        <section class="panel mt-6">
          <h2 class="mb-1 font-semibold">Nube de palabras</h2>
          <p class="subtitulo mb-4">Las {{ nube().length }} palabras más frecuentes; el tamaño es proporcional a la frecuencia.</p>
          <p class="flex flex-wrap items-baseline justify-center gap-x-3 gap-y-1 leading-tight">
            @for (p of nube(); track p.ngrama) {
              <span [style.font-size.rem]="p.tam" [style.opacity]="p.op" [class]="p.color" [title]="p.ngrama + ': ' + p.frecuencia">{{ p.ngrama }}</span>
            }
          </p>
        </section>

        <section class="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Balance de clases">
          <app-kpi etiqueta="Mínimo por clase" [valor]="'' + d.balance['mínimo (train)']" [detalle]="'' + d.balance['clase mínima']" />
          <app-kpi etiqueta="Máximo por clase" [valor]="'' + d.balance['máximo (train)']" [detalle]="'' + d.balance['clase máxima']" />
          <app-kpi etiqueta="Razón máx/mín" [valor]="'' + d.balance['razón máx/mín']" detalle="1 = balance perfecto" />
          <app-kpi etiqueta="Entropía normalizada" [valor]="'' + d.balance['entropía normalizada']" [detalle]="'CV = ' + d.balance['coeficiente de variación']" />
        </section>

        <section id="balance-clases" class="panel mt-6 scroll-mt-24">
          <h2 class="mb-1 font-semibold">Consultas por clase (entrenamiento)</h2>
          <p class="subtitulo mb-4">{{ textoPrueba() }}</p>
          <app-estado [cargando]="c.isLoading()" [error]="c.error()" (reintentar)="c.reload()">
            <app-barras titulo="Consultas por clase" [datos]="paginaBalance()" [max]="maxBalance()" />
            <app-paginador [total]="balanceClases().length" [(pagina)]="pagBal" [(tamano)]="tamBal" [opciones]="[15, 30, 77]" etiqueta="clases, de más a menos ejemplos" ancla="balance-clases" />
          </app-estado>
        </section>
      }
    </app-estado>
  `,
})
export class EdaPagina {
  protected readonly r = httpResource<Eda>(() => '/api/eda');
  protected readonly c = httpResource<Clase[]>(() => '/api/clases');
  protected readonly dec = dec;
  protected readonly ngrama = signal<'palabra' | 'bigrama' | 'trigrama'>('palabra');
  protected tiposNgrama(d: Eda) {
    return [
      { id: 'palabra' as const, texto: `${Math.min(15, d.ngramas.palabra.length)} palabras` },
      { id: 'bigrama' as const, texto: `${d.ngramas.bigrama.length} bigramas` },
      { id: 'trigrama' as const, texto: `${d.ngramas.trigrama.length} trigramas` },
    ];
  }
  protected readonly pagBal = signal(1);
  protected readonly tamBal = signal(15);
  protected readonly paginaBalance = computed(() => pagina(this.balanceClases(), this.pagBal(), this.tamBal()));
  protected readonly maxBalance = computed(() => Math.max(1, ...this.balanceClases().map((b) => b.valor)));
  protected readonly num = num;
  /** Tamaño de cada partición según las estadísticas del EDA guardadas en la base. */
  protected n(particion: string) {
    return this.r.value()?.estadisticas.find((e) => e.metrica === 'n_caracteres' && e.particion === particion)?.count ?? 0;
  }
  protected readonly total = computed(() => this.n('train') + this.n('test'));
  protected readonly textoPrueba = computed(() => {
    const v = [...new Set((this.c.value() ?? []).map((c) => c.n_test))];
    return v.length === 1 ? `En prueba todas las clases tienen exactamente ${v[0]} consultas.` : `En prueba, entre ${Math.min(...v)} y ${Math.max(...v)} consultas por clase.`;
  });

  protected nombreMetrica(m: string) {
    return NOMBRE_METRICA[m] ?? m;
  }
  protected barras(lista: { ngrama: string; frecuencia: number }[], tono: 'acento' | 'bien' | 'mal' = 'acento') {
    return lista.map((n) => ({ etiqueta: n.ngrama, valor: n.frecuencia, texto: num(n.frecuencia), tono }));
  }

  protected readonly nube = computed(() => {
    const p = this.r.value()?.ngramas.palabra ?? [];
    if (!p.length) return [];
    const max = p[0].frecuencia, min = p[p.length - 1].frecuencia;
    const colores = ['text-acento', 'text-bien', 'text-texto', 'text-aviso', 'text-mal'];
    // Orden alfabético estable para que la nube no quede como una lista ordenada por tamaño
    return [...p]
      .sort((a, b) => a.ngrama.localeCompare(b.ngrama))
      .map((w, i) => {
        const f = Math.sqrt((w.frecuencia - min) / Math.max(1, max - min));
        return { ...w, tam: 0.8 + f * 2.4, op: 0.6 + f * 0.4, color: colores[i % colores.length] };
      });
  });

  protected readonly balanceClases = computed(() =>
    [...(this.c.value() ?? [])]
      .sort((a, b) => b.n_train - a.n_train)
      .map((c) => ({ etiqueta: c.nombre, valor: c.n_train, texto: String(c.n_train), tono: 'acento' as const })),
  );
}
