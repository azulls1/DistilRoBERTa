-- ════════════════════════════════════════════════════════════════════════════
-- DistilRoBERTa · endurecimiento (auditoría de seguridad del 26-sep-2026)
-- Idempotente. Reaplica a TODAS las tablas DistilRoBERTa_* lo de 002 (revocar anon/authenticated,
-- RLS forzado, política solo para distilroberta_app) y revoca también las secuencias, que Supabase
-- concede por defecto a anon/authenticated. Toda migración futura que cree tablas debe terminar
-- ejecutando este archivo (los privilegios por defecto del esquema public abren las tablas nuevas).
-- ════════════════════════════════════════════════════════════════════════════
begin;

do $$
declare
    t text;
    s text;
begin
    for t in select tablename from pg_tables where schemaname = 'public' and tablename like 'DistilRoBERTa\_%' loop
        execute format('revoke all on public.%I from anon, authenticated', t);
        execute format('alter table public.%I enable row level security', t);
        execute format('alter table public.%I force row level security', t);
        execute format('grant select, insert, update, delete on public.%I to distilroberta_app', t);
        execute format('drop policy if exists "DistilRoBERTa_app_todo" on public.%I', t);
        execute format('create policy "DistilRoBERTa_app_todo" on public.%I for all to distilroberta_app using (true) with check (true)', t);
    end loop;
    for t in select viewname from pg_views where schemaname = 'public' and viewname like 'DistilRoBERTa\_%' loop
        execute format('revoke all on public.%I from anon, authenticated', t);
    end loop;
    for s in select sequencename from pg_sequences where schemaname = 'public' and sequencename like 'DistilRoBERTa\_%' loop
        execute format('revoke all on sequence public.%I from anon, authenticated', s);
        execute format('grant usage, select on sequence public.%I to distilroberta_app', s);
    end loop;
end $$;

commit;
