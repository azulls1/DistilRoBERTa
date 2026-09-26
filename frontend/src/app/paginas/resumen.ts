import { Component, computed } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { RouterLink } from '@angular/router';
import { Estado } from '../componentes/estado';
import { Kpi } from '../componentes/kpi';
import { Curva } from '../componentes/curva';
import { Epoca, Resumen } from '../core/modelos';
import { num, pct } from '../core/formato';

@Component({
  selector: 'app-resumen',
  imports: [Estado, Kpi, Curva, RouterLink],
  template: `
    <header class="mb-6 grid gap-2">
      <h1 class="titulo">Clasificación de intenciones bancarias</h1>
      <p class="max-w-3xl text-tenue">
        DistilRoBERTa afinado sobre <strong>PolyAI/banking77</strong> para asignar cada consulta de un cliente a una de
        77 intenciones. Los errores del modelo se explican con <strong>Falcon-7b-instruct</strong> usando un prompt calibrado.
        Todas las cifras salen de la ejecución del notebook y se leen de la base de datos.
      </p>
    </header>

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.hasValue()" (reintentar)="r.reload()">
      @if (r.value(); as d) {
        <section class="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Indicadores principales">
          <app-kpi etiqueta="Accuracy en prueba" [valor]="pct(d.corrida.accuracy)" [detalle]="num(d.corrida.n_test) + ' consultas'" />
          <app-kpi etiqueta="F1 macro" [valor]="pct(d.corrida.f1_macro)" detalle="promedio sin ponderar de 77 clases" />
          <app-kpi etiqueta="Errores" [valor]="num(d.n_errores)" [detalle]="'de ' + num(d.corrida.n_test)" />
          <app-kpi etiqueta="Clases" [valor]="num(d.n_clases)" detalle="intenciones bancarias" />
          <app-kpi etiqueta="Entrenamiento" [valor]="num(d.corrida.n_train)" [detalle]="'+ ' + num(d.corrida.n_val) + ' de validación'" />
          <app-kpi etiqueta="Mejor época" [valor]="'' + (d.corrida.hiperparametros['mejor_epoca'] ?? '—')" [detalle]="'de ' + d.corrida.hiperparametros['epocas_max'] + ' máx., early stopping'" />
          <app-kpi etiqueta="Duración" [valor]="minutos(d.corrida.duracion_entrenamiento_s)" [detalle]="'dispositivo: ' + d.corrida.dispositivo" />
          <app-kpi etiqueta="Prompt LLM" [valor]="'Config. ' + (d.corrida.hiperparametros['config_llm'] ?? '—')" [detalle]="temperatura(d)" />
        </section>

        <section class="mt-6 grid gap-6 lg:grid-cols-2">
          <div class="panel">
            <h2 class="mb-1 font-semibold">Curvas de entrenamiento</h2>
            <p class="subtitulo mb-4">Pérdida por época en entrenamiento y validación.</p>
            <app-estado [cargando]="h.isLoading()" [error]="h.error()" (reintentar)="h.reload()">
              <app-curva titulo="Pérdida por época" [epocas]="epocas()" [series]="seriesPerdida()" />
            </app-estado>
          </div>
          <div class="panel">
            <h2 class="mb-1 font-semibold">Métricas en validación</h2>
            <p class="subtitulo mb-4">La mejor época según F1 macro es la que se conserva.</p>
            <app-estado [cargando]="h.isLoading()" [error]="h.error()" (reintentar)="h.reload()">
              <app-curva titulo="Accuracy y F1 macro por época" [epocas]="epocas()" [series]="seriesMetricas()" />
            </app-estado>
          </div>
        </section>

        <section class="panel mt-6">
          <h2 class="mb-3 font-semibold">Cómo se hizo</h2>
          <ol class="grid gap-3 text-sm md:grid-cols-3">
            <li><span class="font-medium">1. Análisis exploratorio.</span> Longitudes, limpieza, n-gramas, nube de palabras y balance de clases. <a routerLink="/eda" class="text-acento underline-offset-2 hover:underline">Ver</a></li>
            <li><span class="font-medium">2. Fine-tuning.</span> {{ d.corrida.modelo_base }}, max_length {{ d.corrida.hiperparametros['max_length'] }}, lr {{ d.corrida.hiperparametros['learning_rate'] }}, lote {{ d.corrida.hiperparametros['lote'] }}. <a routerLink="/clases" class="text-acento underline-offset-2 hover:underline">Ver</a></li>
            <li><span class="font-medium">3. Explicación de errores.</span> {{ d.corrida.llm }} sobre 20 errores, con revisión manual. <a routerLink="/explicaciones" class="text-acento underline-offset-2 hover:underline">Ver</a></li>
          </ol>
        </section>
      }
    </app-estado>
  `,
})
export class ResumenPagina {
  protected readonly r = httpResource<Resumen>(() => '/api/resumen');
  protected readonly h = httpResource<Epoca[]>(() => '/api/entrenamiento');
  protected readonly pct = pct;
  protected readonly num = num;

  protected readonly epocas = computed(() => (this.h.value() ?? []).map((e) => e.epoca));
  protected readonly seriesPerdida = computed(() => {
    const h = this.h.value() ?? [];
    return [
      { nombre: 'train', valores: h.map((e) => e.train_loss), clase: 'text-acento' },
      { nombre: 'validación', valores: h.map((e) => e.eval_loss), clase: 'text-aviso' },
    ];
  });
  protected readonly seriesMetricas = computed(() => {
    const h = this.h.value() ?? [];
    return [
      { nombre: 'accuracy', valores: h.map((e) => e.eval_accuracy), clase: 'text-bien' },
      { nombre: 'F1 macro', valores: h.map((e) => e.eval_f1_macro), clase: 'text-acento' },
    ];
  });

  protected minutos(s: number | null) {
    return s == null ? '—' : `${(s / 60).toFixed(1)} min`;
  }
  protected temperatura(d: Resumen) {
    const t = d.corrida.hiperparametros['llm_temperature'];
    const m = d.corrida.hiperparametros['llm_max_new_tokens'];
    return `${t == null ? 'codiciosa (T→0)' : 'T = ' + t} · ${m} tokens máx.`;
  }
}
