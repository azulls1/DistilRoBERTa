#!/usr/bin/env bash
# Despliega DistilRoBERTa en el Docker Swarm del VPS iaGentek.
#   1) sincroniza el código y el modelo al VPS   2) construye las imágenes con el sha del commit
#   3) crea el secreto del DSN si falta          4) stack deploy + fuerza la imagen nueva
#   5) verifica las tres cosas: especificación, contenedor recién creado y contenido servido
set -euo pipefail

VPS="${VPS:-root@144.126.157.81}"
PUERTO="${PUERTO:-22000}"
LLAVE="${LLAVE:-$HOME/.ssh/id_ed25519}"
DESTINO=/opt/stacks/distilroberta
DOMINIO=distilroberta.iagentek.com.mx
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -p "$PUERTO" -i "$LLAVE" "$VPS")

cd "$RAIZ"
TAG="$(git rev-parse --short=12 HEAD)"
[[ -n "$(git status --porcelain -- backend frontend deploy)" ]] && TAG="${TAG}-dirty"
[[ -f modelo/model.safetensors ]] || { echo "Falta modelo/model.safetensors: ejecuta el notebook primero"; exit 1; }
echo "▶ Desplegando TAG=$TAG"
# Carpeta entregables/ + manifiesto con SHA-256 (se hornea en la imagen del backend)
"$RAIZ/.venv-ml/bin/python" ml/preparar_entregables.py

rsync -az --delete -e "ssh -p $PUERTO -i $LLAVE" \
  --exclude .git --exclude '.venv*' --exclude node_modules --exclude dist --exclude .angular \
  --exclude salidas_entrenamiento --exclude _bmad --exclude _bmad-output --exclude .claude \
  --exclude notebooks --exclude data \
  ./ "$VPS:$DESTINO/src/"

"${SSH[@]}" bash -s -- "$TAG" "$DESTINO" "$DOMINIO" <<'REMOTO'
set -euo pipefail
TAG="$1"; DESTINO="$2"; DOMINIO="$3"
cd "$DESTINO/src"

docker build -q -f backend/Dockerfile -t "distilroberta-backend:$TAG" -t distilroberta-backend:latest .
docker build -q -t "distilroberta-frontend:$TAG" -t distilroberta-frontend:latest frontend

# Secreto con el DSN (la contraseña vive en /root/dados_vps/distilroberta.env, nunca en el repo)
if ! docker secret inspect distilroberta_db_dsn_v1 >/dev/null 2>&1; then
  source /root/dados_vps/distilroberta.env
  printf 'postgresql://distilroberta_app:%s@supabase-maestria_db:5432/postgres' "$DISTILROBERTA_DB_PASSWORD" \
    | docker secret create distilroberta_db_dsn_v1 - >/dev/null
  echo "secreto distilroberta_db_dsn_v1 creado"
fi

TAG="$TAG" docker stack deploy -c deploy/stack.yml --resolve-image=never --detach=false distilroberta
# «converged» no garantiza la imagen nueva: se fuerza explícitamente
for s in api worker; do docker service update -q --detach=false --image "distilroberta-backend:$TAG" "distilroberta_$s" >/dev/null; done
docker service update -q --detach=false --image "distilroberta-frontend:$TAG" distilroberta_web >/dev/null

echo "── Especificación"
for s in api worker web redis; do
  printf '%-22s %s  %s\n' "distilroberta_$s" "$(docker service ls -f name=distilroberta_$s --format '{{.Replicas}}')" \
    "$(docker service inspect distilroberta_$s --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' | cut -d@ -f1)"
done
echo "── Contenedores"
docker ps --filter name=distilroberta_ --format '{{.Names}}  {{.Status}}' | sed 's/\.[a-z0-9]\{25\}//'
echo "── Contenido servido"
sleep 5
curl -s -o /dev/null -w "web  %{http_code}  %{content_type}\n" "https://$DOMINIO/"
curl -s -w "\n" "https://$DOMINIO/api/health"
REMOTO
