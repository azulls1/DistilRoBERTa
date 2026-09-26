import { Component, computed } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Barras } from '../componentes/barras';
import { Kpi } from '../componentes/kpi';
import { Clase, Eda } from '../core/modelos';
import { dec, num } from '../core/formato';

const NOMBRE_METRICA: Record<string, string> = {
  n_caracteres: 'Caracteres',
  n_palabras: 'Palabras',
  n_tokens: 'Tokens (DistilRoBERTa)',
};

@Component({
  selector: 'app-eda',
  imports: [Estado, Barras, Kpi],
  template: `
    <header class="mb-6 grid gap-2">
      <h1 class="titulo">Análisis exploratorio</h1>
      <p class="max-w-3xl text-tenue">
        13 083 consultas en inglés (10 003 de entrenamiento y 3 080 de prueba). Las frecuencias se calculan sobre el
        texto limpio: minúsculas, sin caracteres especiales y sin stopwords.
      </p>
    </header>

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

        <section class="mt-6 grid gap-6 lg:grid-cols-3">
          <div class="panel"><h2 class="mb-3 font-semibold">15 palabras más frecuentes</h2>
            <app-barras titulo="Palabras más frecuentes" [datos]="barras(d.ngramas.palabra.slice(0, 15))" /></div>
          <div class="panel"><h2 class="mb-3 font-semibold">10 bigramas</h2>
            <app-barras titulo="Bigramas más frecuentes" [datos]="barras(d.ngramas.bigrama, 'bien')" /></div>
          <div class="panel"><h2 class="mb-3 font-semibold">10 trigramas</h2>
            <app-barras titulo="Trigramas más frecuentes" [datos]="barras(d.ngramas.trigrama, 'mal')" /></div>
        </section>

        <section class="panel mt-6">
          <h2 class="mb-1 font-semibold">Nube de palabras</h2>
          <p class="subtitulo mb-4">Las 100 palabras más frecuentes; el tamaño es proporcional a la frecuencia.</p>
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

        <section class="panel mt-6">
          <h2 class="mb-1 font-semibold">Consultas por clase (entrenamiento)</h2>
          <p class="subtitulo mb-4">En prueba todas las clases tienen exactamente 40 consultas.</p>
          <app-estado [cargando]="c.isLoading()" [error]="c.error()" (reintentar)="c.reload()">
            <app-barras titulo="Consultas por clase" [datos]="balanceClases()" />
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
