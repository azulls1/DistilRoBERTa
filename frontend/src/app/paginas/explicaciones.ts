import { Component, computed, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { Estado } from '../componentes/estado';
import { Kpi } from '../componentes/kpi';
import { Calibracion, Explicacion } from '../core/modelos';
import { pct } from '../core/formato';

const VEREDICTO = {
  pertinente: { texto: 'Pertinente', clase: 'bg-bien-suave text-bien' },
  parcial: { texto: 'Parcial', clase: 'bg-aviso-suave text-aviso' },
  alucinada: { texto: 'Alucinada', clase: 'bg-mal-suave text-mal' },
} as const;

const RAZON: Record<string, string> = {
  solapamiento_lexico: 'Solapamiento léxico',
  ambiguedad_real: 'Ambigüedad real',
  etiqueta_dudosa: 'Etiqueta dudosa',
  generica: 'Genérica',
  otra: 'Otra',
};

@Component({
  selector: 'app-explicaciones',
  imports: [Estado, Kpi],
  template: `
    <header class="mb-6 grid gap-2">
      <h1 class="titulo">Explicaciones del LLM</h1>
      <p class="max-w-3xl text-tenue">
        Falcon-7b-instruct explica, en máximo dos oraciones, por qué el clasificador se equivocó en las 20 consultas que
        falló con más confianza. Cada explicación se revisó a mano y lleva su veredicto.
      </p>
    </header>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.value()?.length" (reintentar)="r.reload()">
      <section class="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Resumen de la revisión manual">
        @for (k of conteo(); track k.etiqueta) {
          <app-kpi [etiqueta]="k.etiqueta" [valor]="k.valor" [detalle]="k.detalle" />
        }
      </section>

      <div class="mt-6 flex flex-wrap gap-2" role="group" aria-label="Filtrar por veredicto">
        @for (f of filtros; track f.id) {
          <button type="button" class="rounded-full border px-3 py-1 text-sm transition"
                  [class]="filtro() === f.id ? 'border-acento bg-acento-suave text-acento' : 'border-borde hover:bg-borde/40'"
                  [attr.aria-pressed]="filtro() === f.id" (click)="filtro.set(f.id)">{{ f.texto }}</button>
        }
      </div>

      <section class="mt-4 grid gap-4 lg:grid-cols-2">
        @for (e of visibles(); track e.orden) {
          <article class="panel grid gap-3">
            <div class="flex items-start justify-between gap-3">
              <p class="font-medium">“{{ e.texto }}”</p>
              <span class="etiqueta shrink-0" [class]="veredicto(e.veredicto).clase">{{ veredicto(e.veredicto).texto }}</span>
            </div>
            <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
              <dt class="text-tenue">Real</dt><dd class="mono text-bien">{{ e.real }}</dd>
              <dt class="text-tenue">Predicha</dt><dd class="mono text-mal">{{ e.pred }} <span class="text-tenue">({{ pct(e.confianza, 1) }})</span></dd>
              <dt class="text-tenue">Razón</dt><dd>{{ razon(e.razon_categoria) }}</dd>
            </dl>
            <blockquote class="border-l-2 border-acento pl-3 text-sm leading-relaxed">{{ e.explicacion }}</blockquote>
            @if (e.nota_revision) {
              <p class="text-xs text-tenue"><span class="font-medium">Revisión:</span> {{ e.nota_revision }}</p>
            }
          </article>
        } @empty {
          <p class="text-sm text-tenue">No hay explicaciones con ese veredicto.</p>
        }
      </section>

      <section class="panel mt-8">
        <h2 class="mb-1 font-semibold">Calibración del prompt</h2>
        <p class="subtitulo mb-4">
          Tres configuraciones sobre las mismas 5 consultas. A: decodificación codiciosa; B: temperatura 0.3;
          C: temperatura 1.0 con 150 tokens. La elegida se marca.
        </p>
        <app-estado [cargando]="c.isLoading()" [error]="c.error()" (reintentar)="c.reload()">
          <div class="overflow-x-auto">
            <table class="tabla min-w-[48rem]">
              <thead><tr><th>Consulta</th><th>Config.</th><th class="text-right">Oraciones</th><th class="text-right">Tokens</th>
                <th class="text-right">Palabras ajenas</th><th>Salida</th></tr></thead>
              <tbody>
                @for (k of c.value() ?? []; track $index) {
                  <tr [class.bg-acento-suave]="k.elegida">
                    <td class="max-w-[14rem]">{{ k.consulta }}</td>
                    <td class="mono">{{ k.config }}@if (k.elegida) { ✓ }</td>
                    <td class="text-right tabular-nums" [class.text-mal]="k.n_oraciones > 2">{{ k.n_oraciones }}</td>
                    <td class="text-right tabular-nums">{{ k.n_tokens }}</td>
                    <td class="text-right tabular-nums">{{ pct(k.palabras_ajenas, 0) }}</td>
                    <td class="text-xs leading-relaxed">{{ k.salida }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </app-estado>
      </section>

      @if (r.value()?.[0]; as e) {
        <details class="panel mt-6">
          <summary class="cursor-pointer font-semibold">Ver el prompt exacto (ejemplo de la consulta 1)</summary>
          <pre class="mono mt-3 overflow-x-auto whitespace-pre-wrap rounded-lg bg-fondo p-4 text-xs leading-relaxed">{{ e.prompt }}</pre>
        </details>
      }
    </app-estado>
  `,
})
export class ExplicacionesPagina {
  protected readonly r = httpResource<Explicacion[]>(() => '/api/explicaciones');
  protected readonly c = httpResource<Calibracion[]>(() => '/api/calibracion');
  protected readonly pct = pct;
  protected readonly filtro = signal<'todas' | Explicacion['veredicto']>('todas');
  protected readonly filtros = [
    { id: 'todas' as const, texto: 'Todas' },
    { id: 'pertinente' as const, texto: 'Pertinentes' },
    { id: 'parcial' as const, texto: 'Parciales' },
    { id: 'alucinada' as const, texto: 'Alucinadas' },
  ];

  protected readonly visibles = computed(() => {
    const f = this.filtro();
    return (this.r.value() ?? []).filter((e) => f === 'todas' || e.veredicto === f);
  });

  protected readonly conteo = computed(() => {
    const e = this.r.value() ?? [];
    const n = (v: string) => e.filter((x) => x.veredicto === v).length;
    const dos = e.filter((x) => (x.explicacion.match(/[.!?](\s|$)/g) ?? []).length <= 2).length;
    return [
      { etiqueta: 'Pertinentes', valor: `${n('pertinente')} / ${e.length}`, detalle: 'razón plausible y fiel a la consulta' },
      { etiqueta: 'Parciales', valor: `${n('parcial')} / ${e.length}`, detalle: 'algo cierto pero vago' },
      { etiqueta: 'Alucinadas', valor: `${n('alucinada')} / ${e.length}`, detalle: 'afirma lo que la consulta no dice' },
      { etiqueta: '≤ 2 oraciones', valor: `${dos} / ${e.length}`, detalle: 'tras el postproceso' },
    ];
  });

  protected veredicto(v: Explicacion['veredicto']) {
    return VEREDICTO[v] ?? VEREDICTO.parcial;
  }
  protected razon(r: string) {
    return RAZON[r] ?? r;
  }
}
