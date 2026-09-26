-- De dónde sale cada exigencia: el enunciado (docx), la rúbrica detallada de Moodle o una solicitud del alumno
alter table public."DistilRoBERTa_cumplimiento" add column if not exists fuente text not null default 'Enunciado';
