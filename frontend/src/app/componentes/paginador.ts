import { Component, computed, effect, input, model } from '@angular/core';

/**
 * Paginación accesible para listados largos: «x–y de N», selector de tamaño, anterior/siguiente y
 * números de página con elipsis (en móvil solo «página x de y»). Al cambiar de página, desplaza la
 * vista al inicio del listado indicado en `ancla` si quedó por encima de la pantalla.
 */
@Component({
  selector: 'app-paginador',
  template: `
    @if (total() > opciones()[0] || paginas() > 1) {
      <nav class="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm" [attr.aria-label]="'Paginación de ' + etiqueta()">
        <p class="font-mono text-[11px] text-moss tabular-nums" aria-live="polite">
          {{ desde() }}–{{ hasta() }} de {{ total() }} {{ etiqueta() }}
        </p>
        <div class="flex items-center gap-1">
          <button type="button" class="pag-btn" (click)="ir(pagina() - 1)" [disabled]="pagina() <= 1" aria-label="Página anterior">
            <svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2"><path d="m15 18-6-6 6-6" /></svg>
          </button>
          <span class="px-2 font-mono text-xs tabular-nums sm:hidden">{{ pagina() }} / {{ paginas() }}</span>
          @for (p of numeros(); track $index) {
            @if (p === 0) {
              <span class="hidden w-6 text-center text-moss sm:inline" aria-hidden="true">…</span>
            } @else {
              <button type="button" class="pag-btn hidden tabular-nums sm:inline-flex" [class.pag-btn--activa]="p === pagina()"
                      [attr.aria-current]="p === pagina() ? 'page' : null" [attr.aria-label]="'Página ' + p" (click)="ir(p)">{{ p }}</button>
            }
          }
          <button type="button" class="pag-btn" (click)="ir(pagina() + 1)" [disabled]="pagina() >= paginas()" aria-label="Página siguiente">
            <svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6" /></svg>
          </button>
        </div>
        @if (opciones().length > 1) {
          <label class="flex items-center gap-2 text-xs text-tenue">Por página
            <select class="campo !w-auto !py-1 text-xs" [value]="tamano()" (change)="cambiarTamano(+$any($event.target).value)">
              @for (o of opciones(); track o) { <option [value]="o">{{ o }}</option> }
            </select>
          </label>
        }
      </nav>
    }
  `,
  styles: `
    .pag-btn { display: inline-flex; height: 2rem; min-width: 2rem; align-items: center; justify-content: center; border-radius: 0.5rem;
      border: 1px solid transparent; padding: 0 0.5rem; color: #04202C; transition: background-color .15s, border-color .15s; }
    .pag-btn:hover:not(:disabled) { background: #EEF2EF; border-color: #DFE4E0; }
    .pag-btn:disabled { opacity: .35; cursor: not-allowed; }
    .pag-btn:focus-visible { outline: 2px solid #04202C; outline-offset: 2px; }
    .pag-btn--activa, .pag-btn--activa:hover:not(:disabled) { background: #04202C; color: #fff; border-color: #04202C; }
  `,
})
export class Paginador {
  readonly total = input.required<number>();
  readonly pagina = model(1);
  readonly tamano = model(10);
  readonly opciones = input<number[]>([10, 20, 50]);
  readonly etiqueta = input('elementos');
  /** id del elemento al que se vuelve al cambiar de página */
  readonly ancla = input<string>('');

  protected readonly paginas = computed(() => Math.max(1, Math.ceil(this.total() / this.tamano())));
  protected readonly desde = computed(() => (this.total() ? (this.pagina() - 1) * this.tamano() + 1 : 0));
  protected readonly hasta = computed(() => Math.min(this.total(), this.pagina() * this.tamano()));
  /** Números visibles; 0 = elipsis. */
  protected readonly numeros = computed(() => {
    const n = this.paginas(), a = this.pagina();
    if (n <= 7) return Array.from({ length: n }, (_, i) => i + 1);
    const s = new Set([1, n, a - 1, a, a + 1].filter((p) => p >= 1 && p <= n));
    const orden = [...s].sort((x, y) => x - y);
    return orden.flatMap((p, i) => (i && p - orden[i - 1] > 1 ? [0, p] : [p]));
  });

  constructor() {
    // Si cambia el total (p. ej. al filtrar) y la página quedó fuera de rango, volver a la última válida
    effect(() => {
      if (this.pagina() > this.paginas()) this.pagina.set(this.paginas());
    });
  }

  protected ir(p: number) {
    if (p < 1 || p > this.paginas() || p === this.pagina()) return;
    this.pagina.set(p);
    const el = this.ancla() ? document.getElementById(this.ancla()) : null;
    if (el && el.getBoundingClientRect().top < 0) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  protected cambiarTamano(t: number) {
    this.tamano.set(t);
    this.pagina.set(1);
  }
}

/** Rebanada de la página actual. */
export function pagina<T>(lista: T[], p: number, tam: number): T[] {
  return lista.slice((p - 1) * tam, p * tam);
}
