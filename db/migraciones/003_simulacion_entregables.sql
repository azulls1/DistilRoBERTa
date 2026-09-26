-- ════════════════════════════════════════════════════════════════════════════
-- DistilRoBERTa · simulación, cumplimiento del enunciado y entregables
-- Tras aplicarla, volver a correr 002_seguridad.sql (activa RLS y permisos en las tablas nuevas).
-- ════════════════════════════════════════════════════════════════════════════
begin;

-- Cada exigencia de SCA_individual.docx y dónde se cumple
create table if not exists public."DistilRoBERTa_cumplimiento" (
    orden            smallint primary key,
    criterio         text not null,
    criterio_nombre  text not null,
    puntos           numeric,
    peso             smallint,
    requisito        text not null,
    seccion_notebook text not null,
    ruta_web         text not null,
    evidencia        text not null,
    cumplido         boolean not null
);

-- Salidas reales de Falcon-7b-instruct: configuración B para los 208 errores y A/B/C para los 20 revisados
create table if not exists public."DistilRoBERTa_simulacion_llm" (
    corrida_id    text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    consulta_id   int not null references public."DistilRoBERTa_consultas"(id),
    config        text not null check (config in ('A', 'B', 'C')),
    salida_cruda  text not null,
    explicacion   text not null,
    n_oraciones   smallint not null,
    segundos      numeric not null,
    revisada      boolean not null default false,
    primary key (corrida_id, consulta_id, config)
);

-- Catálogo de archivos entregables (con hash)
create table if not exists public."DistilRoBERTa_entregables" (
    id          serial primary key,
    corrida_id  text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    criterio    text not null,
    tipo        text not null,
    nombre      text not null,
    detalle     text not null,
    ruta        text not null,
    bytes       bigint not null,
    sha256      text not null,
    unique (corrida_id, ruta)
);

-- Paquetes ZIP generados desde la web (tarea de Celery)
create table if not exists public."DistilRoBERTa_entregables_generados" (
    id             uuid primary key default gen_random_uuid(),
    task_id        text unique,
    estado         text not null default 'pendiente' check (estado in ('pendiente', 'completada', 'error')),
    archivo        text,
    bytes          bigint,
    sha256         text,
    n_archivos     int,
    error          text,
    creado_en      timestamptz not null default now(),
    completado_en  timestamptz
);

-- La simulación muestra el texto limpio y los tokens de cada inferencia
alter table public."DistilRoBERTa_inferencias" add column if not exists texto_limpio text;
alter table public."DistilRoBERTa_inferencias" add column if not exists tokens jsonb;

commit;
