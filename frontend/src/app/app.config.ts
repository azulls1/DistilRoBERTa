import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { HttpInterceptorFn, provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { NavigationError, provideRouter, withComponentInputBinding, withInMemoryScrolling, withNavigationErrorHandler } from '@angular/router';
import { routes } from './app.routes';

/**
 * Tras un despliegue, una pestaña abierta con la versión anterior pide archivos de página (chunks) que ya
 * no existen y la navegación fallaría en silencio. Se recarga la página de destino una sola vez, lo que
 * trae la versión nueva; la marca en sessionStorage evita un bucle si el fallo fuera de otra índole.
 */
function recargarSiFaltaVersion(e: NavigationError) {
  const texto = String((e.error as Error)?.message ?? e.error);
  if (!/dynamically imported module|Importing a module script failed|error loading dynamically|ChunkLoadError/i.test(texto)) return;
  try {
    const clave = 'recarga-version:' + e.url;
    if (sessionStorage.getItem(clave)) return;
    sessionStorage.setItem(clave, '1');
  } catch { /* sin almacenamiento: se recarga igual */ }
  location.assign(e.url);
}

/** Cabecera propia en las llamadas a la API: el backend la exige en las acciones que escriben (anti-CSRF). */
const cabeceraPortal: HttpInterceptorFn = (req, next) =>
  next(req.url.startsWith('/api/') ? req.clone({ setHeaders: { 'X-Requested-With': 'portal-distilroberta' } }) : req);

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideHttpClient(withFetch(), withInterceptors([cabeceraPortal])),
    provideRouter(routes, withComponentInputBinding(), withInMemoryScrolling({ scrollPositionRestoration: 'top' }), withNavigationErrorHandler(recargarSiFaltaVersion)),
  ],
};
