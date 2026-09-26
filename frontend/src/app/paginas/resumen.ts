import { Component, computed } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { RouterLink } from '@angular/router';
import { Estado } from '../componentes/estado';
import { Kpi } from '../componentes/kpi';
import { Icono, NombreIcono } from '../componentes/icono';
import { Narracion } from '../componentes/narracion';
import { NARRACIONES } from '../core/narraciones';
import { Curva } from '../componentes/curva';
import { Epoca, Resumen } from '../core/modelos';
import { num, pct } from '../core/formato';

@Component({
  selector: 'app-resumen',
  imports: [Estado, Kpi, Curva, RouterLink, Icono, Narracion],
  template: `
    <section class="card-hero mb-8" aria-labelledby="titulo-hero">
      <div class="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        <div class="orbe absolute -right-24 -top-24 h-80 w-80 rounded-full bg-[#5B7065]/25 blur-3xl"></div>
        <div class="orbe orbe--lento absolute -bottom-32 -left-16 h-72 w-72 rounded-full bg-[#9EADA3]/15 blur-3xl"></div>
        <div class="absolute inset-0 opacity-[0.04]" style="background-image: radial-gradient(circle, white 1px, transparent 1px); background-size: 24px 24px;"></div>
      </div>
      <div class="relative">
        <div class="mb-5 flex flex-wrap justify-center gap-2">
          <span class="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] font-medium tracking-wide text-white/85">
            <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400"></span> Actividad 2 · individual
          </span>
          <span class="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] text-white/85">Sistemas Cognitivos Artificiales</span>
          <span class="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] text-white/85">Maestría en IA · UNIR 2026</span>
        </div>
        <h1 id="titulo-hero" class="font-display text-3xl font-bold tracking-tight text-white sm:text-4xl md:text-5xl">
          Transformers y LLM
          <span class="mt-2 block text-xl font-medium text-white/70 sm:text-2xl">DistilRoBERTa + Falcon-7b-instruct</span>
        </h1>
        <p class="mx-auto mt-4 max-w-2xl text-sm leading-relaxed text-white/75 sm:text-base">
          Clasificación de consultas bancarias de <strong class="text-white">PolyAI/banking77</strong> en 77 intenciones con un
          Transformer afinado, y explicación de sus errores con un LLM mediante un prompt calibrado.
        </p>
        <p class="mx-auto mt-2 max-w-xl text-xs text-white/50">Todas las cifras salen de la ejecución del notebook y se leen de la base de datos.</p>
        <p class="mt-5 text-base font-semibold text-white">Adonai Samael Hernández Mata</p>
        <div class="mt-6 flex flex-wrap justify-center gap-3">
          <a routerLink="/simulacion" class="inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-sm font-medium text-forest transition hover:-translate-y-px"><app-icono nombre="simulacion" clase="h-4 w-4" /> Probar la simulación</a>
          <a routerLink="/entregables" class="inline-flex items-center gap-2 rounded-lg border border-white/25 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10"><app-icono nombre="descarga" clase="h-4 w-4" /> Entregables</a>
          <a href="https://github.com/azulls1/DistilRoBERTa" target="_blank" rel="noopener"
             class="inline-flex items-center gap-2 rounded-lg border border-white/25 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10"><app-icono nombre="github" clase="h-4 w-4" /> GitHub</a>
        </div>
      </div>
    </section>

    <app-narracion class="mb-8 block animate-fadeInUp" [src]="narracion.src" [titulo]="narracion.titulo" [transcripcion]="narracion.texto" />

    <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.hasValue()" (reintentar)="r.reload()">
      @if (r.value(); as d) {
        <h2 class="seccion">Resultados en prueba</h2>
        <section class="stagger-children grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Indicadores principales">
          <app-kpi icono="diana" etiqueta="Accuracy en prueba" [valor]="pct(d.corrida.accuracy)" [detalle]="num(d.corrida.n_test) + ' consultas'" />
          <app-kpi icono="grafica" etiqueta="F1 macro" [valor]="pct(d.corrida.f1_macro)" [detalle]="'promedio sin ponderar de ' + d.n_clases + ' clases'" />
          <app-kpi icono="alerta" etiqueta="Errores" [valor]="num(d.n_errores)" [detalle]="'de ' + num(d.corrida.n_test)" />
          <app-kpi icono="etiqueta" etiqueta="Clases" [valor]="num(d.n_clases)" detalle="intenciones bancarias" />
          <app-kpi icono="datos" etiqueta="Entrenamiento" [valor]="num(d.corrida.n_train)" [detalle]="'+ ' + num(d.corrida.n_val) + ' de validación'" />
          <app-kpi icono="reiniciar" etiqueta="Mejor época" [valor]="'' + (d.corrida.hiperparametros['mejor_epoca'] ?? '—')" [detalle]="'de ' + d.corrida.hiperparametros['epocas_max'] + ' máx., early stopping'" />
          <app-kpi icono="reloj" etiqueta="Duración" [valor]="minutos(d.corrida.duracion_entrenamiento_s)" [detalle]="'dispositivo: ' + d.corrida.dispositivo" />
          <app-kpi icono="termometro" etiqueta="Prompt LLM" [valor]="'Config. ' + (d.corrida.hiperparametros['config_llm'] ?? '—')" [detalle]="temperatura(d)" />
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

        <h2 class="seccion mt-8">Explora la actividad</h2>
        <section class="stagger-children grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          @for (t of tarjetas; track t.ruta) {
            <a [routerLink]="t.ruta" class="panel group flex flex-col gap-3 transition duration-300 hover:-translate-y-1">
              <span class="flex h-10 w-10 items-center justify-center rounded-lg bg-acento-suave text-pine transition group-hover:bg-forest group-hover:text-white">
                <app-icono [nombre]="t.icono" clase="h-5 w-5" /></span>
              <span class="font-mono text-[10px] uppercase tracking-[0.12em] text-moss">{{ t.criterio }}</span>
              <span class="font-display font-semibold text-forest">{{ t.titulo }}</span>
              <span class="text-sm text-tenue">{{ t.texto }}</span>
              <span class="mt-auto inline-flex items-center gap-1 text-xs font-medium text-pine">Abrir <app-icono nombre="flecha" clase="h-3.5 w-3.5 transition group-hover:translate-x-1" /></span>
            </a>
          }
        </section>
      }
    </app-estado>
  `,
})
export class ResumenPagina {
  protected readonly narracion = NARRACIONES.recorrido;
  protected readonly tarjetas: { ruta: string; icono: NombreIcono; criterio: string; titulo: string; texto: string }[] = [
    { ruta: '/eda', icono: 'grafica', criterio: 'C1 · 15 %', titulo: 'Análisis exploratorio', texto: 'Longitudes, limpieza, n-gramas, nube de palabras y balance de las 77 clases.' },
    { ruta: '/clases', icono: 'lista', criterio: 'C2 · 25 %', titulo: 'Transformer', texto: 'Fine-tuning de DistilRoBERTa, métricas por clase y las 7 mejores y 7 peores.' },
    { ruta: '/explicaciones', icono: 'mensaje', criterio: 'C3 · 30 %', titulo: 'Prompt con Falcon-7b', texto: 'Calibración de temperatura, longitud y estructura del prompt; explicaciones de los errores y su revisión manual.' },
    { ruta: '/simulacion', icono: 'simulacion', criterio: 'En vivo', titulo: 'Simulación', texto: 'Una consulta real, paso a paso: tokens, 6 capas, top-5 y la respuesta de Falcon.' },
  ];
  protected readonly r = httpResource<Resumen>(() => '/api/resumen');
  protected readonly h = httpResource<Epoca[]>(() => '/api/entrenamiento');
  protected readonly pct = pct;
  protected readonly num = num;

  protected readonly epocas = computed(() => (this.h.value() ?? []).map((e) => e.epoca));
  protected readonly seriesPerdida = computed(() => {
    const h = this.h.value() ?? [];
    return [
      { nombre: 'train', valores: h.map((e) => e.train_loss), clase: 'text-forest' },
      { nombre: 'validación', valores: h.map((e) => e.eval_loss), clase: 'text-moss' },
    ];
  });
  protected readonly seriesMetricas = computed(() => {
    const h = this.h.value() ?? [];
    return [
      { nombre: 'accuracy', valores: h.map((e) => e.eval_accuracy), clase: 'text-pine' },
      { nombre: 'F1 macro', valores: h.map((e) => e.eval_f1_macro), clase: 'text-forest' },
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
