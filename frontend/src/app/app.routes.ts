import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'resumen' },
  { path: 'resumen', title: 'Resumen · DistilRoBERTa', loadComponent: () => import('./paginas/resumen').then((m) => m.ResumenPagina) },
  { path: 'eda', title: 'Análisis exploratorio · DistilRoBERTa', loadComponent: () => import('./paginas/eda').then((m) => m.EdaPagina) },
  { path: 'clases', title: 'Desempeño por clase · DistilRoBERTa', loadComponent: () => import('./paginas/clases').then((m) => m.ClasesPagina) },
  { path: 'confusion', title: 'Matriz de confusión · DistilRoBERTa', loadComponent: () => import('./paginas/confusion').then((m) => m.ConfusionPagina) },
  { path: 'explicaciones', title: 'Explicaciones del LLM · DistilRoBERTa', loadComponent: () => import('./paginas/explicaciones').then((m) => m.ExplicacionesPagina) },
  { path: 'clasificar', title: 'Clasificar una consulta · DistilRoBERTa', loadComponent: () => import('./paginas/clasificar').then((m) => m.ClasificarPagina) },
  { path: '**', redirectTo: 'resumen' },
];
