-- ════════════════════════════════════════════════════════════════════════════
-- DistilRoBERTa · Banking77 — esquema (Supabase maestría, esquema public)
-- Nomenclatura: "DistilRoBERTa_<tabla>" (identificadores entre comillas dobles).
-- Idempotente: se puede volver a correr sin perder datos.
-- ════════════════════════════════════════════════════════════════════════════
begin;

-- Una ejecución completa del notebook; la web muestra la que tiene activa = true
create table if not exists public."DistilRoBERTa_corridas" (
    id                        text primary key,
    modelo_base               text not null,
    llm                       text not null,
    dispositivo               text not null,
    hiperparametros           jsonb not null default '{}'::jsonb,
    accuracy                  numeric(6,4) not null check (accuracy between 0 and 1),
    f1_macro                  numeric(6,4) not null check (f1_macro between 0 and 1),
    n_train                   int not null,
    n_val                     int not null,
    n_test                    int not null,
    duracion_entrenamiento_s  numeric,
    activa                    boolean not null default false,
    creado_en                 timestamptz not null default now()
);
create unique index if not exists "DistilRoBERTa_corridas_una_activa"
    on public."DistilRoBERTa_corridas" (activa) where activa;

-- Las 77 intenciones, en el orden oficial del dataset
create table if not exists public."DistilRoBERTa_clases" (
    id              smallint primary key check (id between 0 and 76),
    nombre          text not null unique,
    nombre_legible  text not null,
    n_train         int not null,
    n_test          int not null
);

-- Las 13 083 consultas del dataset (train: id 0…10002 · test: 100000 + índice)
create table if not exists public."DistilRoBERTa_consultas" (
    id            int primary key,
    particion     text not null check (particion in ('train', 'test')),
    texto         text not null,
    texto_limpio  text not null,
    clase_id      smallint not null references public."DistilRoBERTa_clases"(id),
    n_caracteres  int not null,
    n_palabras    int not null,
    n_tokens      int not null
);
create index if not exists "DistilRoBERTa_consultas_clase" on public."DistilRoBERTa_consultas" (clase_id, particion);

create table if not exists public."DistilRoBERTa_eda_estadisticas" (
    corrida_id  text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    metrica     text not null,
    particion   text not null,
    count       int not null,
    mean        numeric not null,
    std         numeric not null,
    min         numeric not null,
    q1          numeric not null,
    mediana     numeric not null,
    q3          numeric not null,
    max         numeric not null,
    primary key (corrida_id, metrica, particion)
);

create table if not exists public."DistilRoBERTa_eda_balance" (
    corrida_id  text primary key references public."DistilRoBERTa_corridas"(id) on delete cascade,
    datos       jsonb not null
);

create table if not exists public."DistilRoBERTa_ngramas" (
    corrida_id  text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    tipo        text not null check (tipo in ('palabra', 'bigrama', 'trigrama')),
    rango       smallint not null,
    ngrama      text not null,
    frecuencia  int not null,
    primary key (corrida_id, tipo, rango)
);

create table if not exists public."DistilRoBERTa_entrenamiento" (
    corrida_id     text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    epoca          smallint not null,
    train_loss     numeric not null,
    eval_loss      numeric not null,
    eval_accuracy  numeric not null,
    eval_f1_macro  numeric not null,
    primary key (corrida_id, epoca)
);

create table if not exists public."DistilRoBERTa_metricas_clase" (
    corrida_id  text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    clase_id    smallint not null references public."DistilRoBERTa_clases"(id),
    precision   numeric not null,
    recall      numeric not null,
    f1          numeric not null,
    soporte     int not null,
    errores     int not null,
    categoria   text check (categoria in ('mejor', 'peor')),
    primary key (corrida_id, clase_id)
);

-- Matriz de confusión dispersa (solo celdas con conteo > 0)
create table if not exists public."DistilRoBERTa_confusion" (
    corrida_id     text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    clase_real_id  smallint not null references public."DistilRoBERTa_clases"(id),
    clase_pred_id  smallint not null references public."DistilRoBERTa_clases"(id),
    conteo         int not null check (conteo > 0),
    primary key (corrida_id, clase_real_id, clase_pred_id)
);

create table if not exists public."DistilRoBERTa_predicciones" (
    corrida_id     text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    consulta_id    int not null references public."DistilRoBERTa_consultas"(id),
    clase_pred_id  smallint not null references public."DistilRoBERTa_clases"(id),
    confianza      numeric(6,5) not null,
    correcta       boolean not null,
    top5           jsonb not null,
    primary key (corrida_id, consulta_id)
);
create index if not exists "DistilRoBERTa_predicciones_errores"
    on public."DistilRoBERTa_predicciones" (corrida_id) where not correcta;

create table if not exists public."DistilRoBERTa_calibracion_llm" (
    id           serial primary key,
    corrida_id   text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    config       text not null,
    parametros   jsonb not null,
    consulta_id  int not null references public."DistilRoBERTa_consultas"(id),
    salida       text not null,
    n_oraciones  smallint not null,
    n_tokens     smallint not null,
    segundos     numeric not null,
    palabras_ajenas numeric,
    elegida      boolean not null default false
);

create table if not exists public."DistilRoBERTa_explicaciones_llm" (
    id               serial primary key,
    corrida_id       text not null references public."DistilRoBERTa_corridas"(id) on delete cascade,
    orden            smallint not null check (orden between 1 and 20),
    consulta_id      int not null references public."DistilRoBERTa_consultas"(id),
    clase_real_id    smallint not null references public."DistilRoBERTa_clases"(id),
    clase_pred_id    smallint not null references public."DistilRoBERTa_clases"(id),
    confianza        numeric(6,5) not null,
    prompt           text not null,
    parametros       jsonb not null,
    salida_cruda     text not null,
    explicacion      text not null,
    razon_categoria  text not null,
    veredicto        text not null check (veredicto in ('pertinente', 'parcial', 'alucinada')),
    nota_revision    text,
    unique (corrida_id, orden),
    check (clase_real_id <> clase_pred_id)
);

-- Consultas enviadas desde la web y clasificadas por el worker de Celery
create table if not exists public."DistilRoBERTa_inferencias" (
    id             uuid primary key default gen_random_uuid(),
    task_id        text unique,
    texto          text not null check (char_length(texto) between 1 and 512),
    estado         text not null default 'pendiente' check (estado in ('pendiente', 'completada', 'error')),
    clase_pred_id  smallint references public."DistilRoBERTa_clases"(id),
    confianza      numeric(6,5),
    top5           jsonb,
    error          text,
    duracion_ms    int,
    creado_en      timestamptz not null default now(),
    completado_en  timestamptz
);
create index if not exists "DistilRoBERTa_inferencias_recientes" on public."DistilRoBERTa_inferencias" (creado_en desc);

-- Resumen de la corrida activa (panel principal)
create or replace view public."DistilRoBERTa_v_resumen" with (security_invoker = true) as
select c.*,
       (select count(*) from public."DistilRoBERTa_clases")                               as n_clases,
       (select count(*) from public."DistilRoBERTa_predicciones" p
         where p.corrida_id = c.id and not p.correcta)                                     as n_errores
from public."DistilRoBERTa_corridas" c
where c.activa;

commit;
