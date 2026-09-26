import { Component, input, output } from '@angular/core';

/**
 * Envoltorio de estados de un recurso remoto: carga, error con reintento, vacío o contenido.
 * Uso: <app-estado [cargando]="r.isLoading()" [error]="r.error()" [vacio]="!r.hasValue()" (reintentar)="r.reload()">…</app-estado>
 */
@Component({
  selector: 'app-estado',
  template: `
    @if (error()) {
      <div class="panel flex flex-col items-start gap-3 border-mal/40 bg-mal-suave" role="alert">
        <p class="font-medium text-mal">No se pudieron cargar los datos.</p>
        <p class="text-sm text-tenue">{{ mensaje() }}</p>
        <button type="button" class="boton" (click)="reintentar.emit()">Reintentar</button>
      </div>
    } @else if (cargando()) {
      <div class="grid gap-3" aria-busy="true" aria-live="polite">
        <span class="sr-only">Cargando…</span>
        @for (i of [1, 2, 3]; track i) {
          <div class="h-20 animate-pulse rounded-xl bg-borde/60"></div>
        }
      </div>
    } @else if (vacio()) {
      <div class="panel text-sm text-tenue">{{ textoVacio() }}</div>
    } @else {
      <ng-content />
    }
  `,
})
export class Estado {
  readonly cargando = input(false);
  readonly error = input<unknown>(null);
  readonly vacio = input(false);
  readonly textoVacio = input('Todavía no hay datos cargados.');
  readonly reintentar = output<void>();

  protected mensaje() {
    const e = this.error() as { status?: number; error?: { detail?: string }; message?: string } | null;
    if (!e) return '';
    if (e.status === 0) return 'El servidor no responde. Revisa tu conexión.';
    return e.error?.detail ?? e.message ?? 'Error desconocido.';
  }
}
