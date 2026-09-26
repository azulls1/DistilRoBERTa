import { Component, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
})
export class App {
  protected readonly menuAbierto = signal(false);
  protected readonly enlaces = [
    { ruta: '/resumen', texto: 'Resumen' },
    { ruta: '/eda', texto: 'Análisis exploratorio' },
    { ruta: '/clases', texto: 'Desempeño por clase' },
    { ruta: '/confusion', texto: 'Matriz de confusión' },
    { ruta: '/explicaciones', texto: 'Explicaciones del LLM' },
    { ruta: '/clasificar', texto: 'Clasificar en vivo' },
  ];
}
