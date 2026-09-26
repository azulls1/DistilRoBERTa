import { Component, ElementRef, computed, input, signal, viewChild } from '@angular/core';
import { Icono } from './icono';

/** Reproductor de las narraciones generadas con ElevenLabs, con transcripción accesible. */
@Component({
  selector: 'app-narracion',
  imports: [Icono],
  template: `
    <section class="panel flex flex-col gap-3 !p-4" [attr.aria-label]="'Narración: ' + titulo()">
      @if (src()) {
      <div class="flex items-center gap-4">
        <button type="button" (click)="alternar()"
                class="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-forest text-white transition hover:scale-105 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-forest"
                [attr.aria-label]="sonando() ? 'Pausar narración' : 'Reproducir narración'">
          @if (sonando()) { <span class="absolute inset-0 animate-ping rounded-full bg-forest/30"></span> }
          <app-icono [nombre]="sonando() ? 'pausa' : 'reproducir'" clase="h-5 w-5" [grosor]="2" />
        </button>
        <div class="min-w-0 flex-1">
          <div class="flex items-baseline justify-between gap-2">
            <p class="truncate font-display text-sm font-semibold text-forest">{{ titulo() }}</p>
            <span class="shrink-0 font-mono text-[11px] text-moss tabular-nums">{{ reloj(actual()) }} / {{ reloj(duracion()) }}</span>
          </div>
          <input type="range" min="0" [max]="duracion() || 1" step="0.1" [value]="actual()" (input)="buscar($event)"
                 class="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-fog accent-forest"
                 [style.background]="'linear-gradient(to right, #04202C ' + progreso() + '%, #DFE4E0 ' + progreso() + '%)'"
                 aria-label="Posición de la narración" />
          <p class="mt-1.5 flex items-center gap-1.5 font-mono text-[10px] text-moss">
            <app-icono nombre="audio" clase="h-3 w-3" /> Voz generada con ElevenLabs
            <button type="button" class="ml-auto text-pine underline-offset-2 hover:underline" (click)="verTexto.set(!verTexto())"
                    [attr.aria-expanded]="verTexto()">{{ verTexto() ? 'Ocultar texto' : 'Ver texto' }}</button>
          </p>
        </div>
      </div>
      } @else {
        <p class="flex items-center gap-2 font-display text-sm font-semibold text-forest"><app-icono nombre="audio" clase="h-4 w-4" /> {{ titulo() }}</p>
      }
      @if (verTexto() || !src()) {
        <p class="animate-fadeIn border-t border-fog/60 pt-3 text-sm leading-relaxed text-evergreen">{{ transcripcion() }}</p>
      }
      @if (src()) {
      <audio #audio [src]="src()" preload="metadata" (timeupdate)="actual.set($any($event.target).currentTime)"
             (loadedmetadata)="duracion.set($any($event.target).duration)" (play)="sonando.set(true)"
             (pause)="sonando.set(false)" (ended)="sonando.set(false)"></audio>
      }
    </section>
  `,
})
export class Narracion {
  readonly src = input.required<string>();
  readonly titulo = input.required<string>();
  readonly transcripcion = input('');
  protected readonly sonando = signal(false);
  protected readonly actual = signal(0);
  protected readonly duracion = signal(0);
  protected readonly verTexto = signal(false);
  protected readonly progreso = computed(() => (this.duracion() ? (this.actual() / this.duracion()) * 100 : 0));
  private readonly audio = viewChild<ElementRef<HTMLAudioElement>>('audio');

  protected alternar() {
    const a = this.audio()?.nativeElement;
    if (!a) return;
    a.paused ? a.play() : a.pause();
  }
  protected buscar(ev: Event) {
    const a = this.audio()?.nativeElement;
    if (a) a.currentTime = Number((ev.target as HTMLInputElement).value);
  }
  protected reloj(s: number) {
    if (!isFinite(s)) return '0:00';
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  }
}
