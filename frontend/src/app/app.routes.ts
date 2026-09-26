import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'resumen' },
  { path: 'resumen', title: 'Resumen · DistilRoBERTa', loadComponent: () => import('./paginas/resumen').then((m) => m.ResumenPagina) },
  { path: 'eda', title: 'Análisis exploratorio · DistilRoBERTa', loadComponent: () => import('./paginas/eda').then((m) => m.EdaPagina) },
  { path: 'clases', title: 'Desempeño por clase · DistilRoBERTa', loadComponent: () => import('./paginas/clases').then((m) => m.ClasesPagina) },
  { path: 'confusion', title: 'Matriz de confusión · DistilRoBERTa', loadComponent: () => import('./paginas/confusion').then((m) => m.ConfusionPagina) },
  { path: 'explicaciones', title: 'Explicaciones del LLM · DistilRoBERTa', loadComponent: () => import('./paginas/explicaciones').then((m) => m.ExplicacionesPagina) },
  { path: 'simulacion', title: 'Simulación · DistilRoBERTa', loadComponent: () => import('./paginas/simulacion').then((m) => m.SimulacionPagina) },
  { path: 'cumplimiento', title: 'Cumplimiento del enunciado · DistilRoBERTa', loadComponent: () => import('./paginas/cumplimiento').then((m) => m.CumplimientoPagina) },
  { path: 'entregables', title: 'Entregables · DistilRoBERTa', loadComponent: () => import('./paginas/entregables').then((m) => m.EntregablesPagina) },
  { path: 'clasificar', title: 'Clasificar una consulta · DistilRoBERTa', loadComponent: () => import('./paginas/clasificar').then((m) => m.ClasificarPagina) },
  { path: '**', redirectTo: 'resumen' },
];
