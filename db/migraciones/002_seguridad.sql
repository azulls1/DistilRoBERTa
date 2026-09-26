-- ════════════════════════════════════════════════════════════════════════════
-- DistilRoBERTa · seguridad
-- 1) Rol propio de la aplicación (la contraseña se pasa con -v app_password=…)
-- 2) Nadie anónimo lee nada: se revocan los privilegios que Supabase concede por defecto a
--    anon/authenticated sobre tablas nuevas de public, y se activa RLS en todas las tablas.
-- 3) Políticas solo para el rol distilroberta_app.
-- Uso: psql -v app_password="'…'" -f 002_seguridad.sql
-- ════════════════════════════════════════════════════════════════════════════
begin;

do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'distilroberta_app') then
        create role distilroberta_app login;
    end if;
end $$;
alter role distilroberta_app with login password :app_password;
alter role distilroberta_app set search_path = public;
grant usage on schema public to distilroberta_app;

do $$
declare
    t text;
begin
    for t in
        select tablename from pg_tables
        where schemaname = 'public' and tablename like 'DistilRoBERTa\_%'
    loop
        execute format('revoke all on public.%I from anon, authenticated', t);
        execute format('alter table public.%I enable row level security', t);
        execute format('alter table public.%I force row level security', t);
        execute format('grant select, insert, update, delete on public.%I to distilroberta_app', t);
        execute format('drop policy if exists "DistilRoBERTa_app_todo" on public.%I', t);
        execute format('create policy "DistilRoBERTa_app_todo" on public.%I for all to distilroberta_app using (true) with check (true)', t);
    end loop;
end $$;

revoke all on public."DistilRoBERTa_v_resumen" from anon, authenticated;
grant select on public."DistilRoBERTa_v_resumen" to distilroberta_app;
do $$
declare
    s text;
begin
    for s in
        select sequencename from pg_sequences
        where schemaname = 'public' and sequencename like 'DistilRoBERTa\_%'
    loop
        execute format('grant usage, select on sequence public.%I to distilroberta_app', s);
    end loop;
end $$;

commit;
