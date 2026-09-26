-- Calibración v2: cinco configuraciones de decodificación (A–E) y tres estructuras de prompt (P1–P3)
begin;
alter table public."DistilRoBERTa_simulacion_llm" drop constraint if exists "DistilRoBERTa_simulacion_llm_config_check";
alter table public."DistilRoBERTa_simulacion_llm" add column if not exists prompt text not null default 'P1';
alter table public."DistilRoBERTa_calibracion_llm" add column if not exists prompt text not null default 'P1';
alter table public."DistilRoBERTa_calibracion_llm" add column if not exists formato_ok boolean;
alter table public."DistilRoBERTa_calibracion_llm" add column if not exists cita_falsa boolean;
alter table public."DistilRoBERTa_calibracion_llm" add column if not exists cita_verificable boolean;
alter table public."DistilRoBERTa_explicaciones_llm" add column if not exists cita_falsa boolean;
commit;
