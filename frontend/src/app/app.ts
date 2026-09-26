import { Component, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

interface Enlace {
  ruta: string;
  texto: string;
  /** Trazo SVG 24×24 (sin librería de iconos, como pide el Forest DS). */
  icono: string;
  criterio?: string;
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
})
export class App {
  protected readonly menuAbierto = signal(false);
  protected readonly grupos: { titulo: string; enlaces: Enlace[] }[] = [
    {
      titulo: 'Principal',
      enlaces: [{ ruta: '/resumen', texto: 'Resumen', icono: 'M3 11l9-8 9 8M5 10v10h14V10' }],
    },
    {
      titulo: 'Clasificador',
      enlaces: [
        { ruta: '/eda', texto: 'Análisis exploratorio', criterio: 'C1', icono: 'M4 19V5M4 19h16M8 15v-3M12 15V8M16 15v-6' },
        { ruta: '/clases', texto: 'Desempeño por clase', criterio: 'C2', icono: 'M4 6h16M4 12h10M4 18h6' },
        { ruta: '/confusion', texto: 'Matriz de confusión', criterio: 'C2', icono: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z' },
      ],
    },
    {
      titulo: 'LLM',
      enlaces: [
        { ruta: '/explicaciones', texto: 'Explicaciones del LLM', criterio: 'C3', icono: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2zM8 9h8M8 13h5' },
      ],
    },
    {
      titulo: 'En vivo',
      enlaces: [
        { ruta: '/simulacion', texto: 'Simulación', icono: 'M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z' },
        { ruta: '/clasificar', texto: 'Clasificar consulta', icono: 'M13 2L3 14h9l-1 8 10-12h-9z' },
      ],
    },
    {
      titulo: 'Entrega',
      enlaces: [
        { ruta: '/cumplimiento', texto: 'Cumplimiento', criterio: '33/33', icono: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10zM9 12l2 2 4-4' },
        { ruta: '/entregables', texto: 'Entregables', icono: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3' },
      ],
    },
  ];
}
