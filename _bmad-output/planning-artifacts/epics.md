---
stepsCompleted: [requirements-inventory, epic-list, stories]
inputDocuments:
  - ../../../espesificaciones/specs/001-clasificador-intenciones-banking77/spec.md
  - ../../../espesificaciones/specs/001-clasificador-intenciones-banking77/tasks.md
  - architecture.md
---

# DistilRoBERTa · Banking77 - Epic Breakdown

## Overview

Las épicas siguen las historias de usuario de la especificación de Spec Kit (US1–US3) más una de
plataforma. Los requisitos **no se copian**: se referencian por su id (FR-/SC-) y las tareas
detalladas están en `tasks.md` (T001–T039).

## Requirements Inventory

### FR Coverage Map

| Épica | Requisitos | Tareas Spec Kit |
|---|---|---|
| 1 · Entregable académico | FR-001..FR-017, SC-001..SC-004, SC-008 | T009–T023 |
| 2 · Datos y API de lectura | FR-018, FR-019, FR-022, SC-005, SC-007 | T006–T008, T024–T026 |
| 3 · Interfaz web | FR-019, FR-020, FR-021 | T027–T030, T034 |
| 4 · Clasificación en vivo y despliegue | FR-020, FR-021, FR-023, SC-006 | T031–T033, T035–T039 |

## Epic List

1. Entregable académico (notebook + PDF)
2. Datos y API de lectura
3. Interfaz web
4. Clasificación en vivo y despliegue

## Epic 1: Entregable académico

### Story 1.1: Notebook ejecutado de punta a punta
As a estudiante, I want un notebook que cubra 1:1 el enunciado, So that obtenga la nota máxima de la rúbrica.

**Given** el dataset oficial **When** se ejecuta `ml/ejecutar_notebook.py` **Then** todas las celdas terminan sin error **And** accuracy en prueba ≥ 0.90.

### Story 1.2: Revisión manual e interpretación
**Given** las 20 explicaciones **When** se revisan **Then** cada una tiene veredicto y razón en `artefactos/revision_manual.json` **And** cada sección cierra con su interpretación escrita con las cifras reales.

### Story 1.3: PDF
**Given** el notebook ejecutado **When** se exporta **Then** existe `notebooks/…pdf` con todas las gráficas.

## Epic 2: Datos y API de lectura

### Story 2.1: Esquema y seguridad
**Given** `db/migraciones/` **When** se aplican **Then** existen 13 tablas `DistilRoBERTa_*` con RLS forzado **And** `anon` no lee filas.

### Story 2.2: Cargador y endpoints de lectura
**Given** `artefactos/` **When** corre el cargador **Then** la corrida queda activa **And** `/api/resumen` devuelve la accuracy del notebook.

## Epic 3: Interfaz web

### Story 3.1: Páginas de resultados
**Given** la API **When** se abre cada página **Then** muestra datos, estado de carga o error con reintento.

## Epic 4: Clasificación en vivo y despliegue

### Story 4.1: Worker Celery y página Clasificar
**Given** una consulta válida **When** se envía **Then** en < 5 s se ve clase, confianza y top-5 **And** queda en el historial.

### Story 4.2: Despliegue en distilroberta.iagentek.com.mx
**Given** `deploy/deploy.sh` **When** se ejecuta **Then** los 4 servicios están 1/1 con la imagen `:<sha>` **And** el dominio responde con TLS válido.
