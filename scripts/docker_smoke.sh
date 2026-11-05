#!/bin/sh
# Build the image and prove it serves. Runs locally and in CI. Removes what it starts.
set -eu

IMAGE=phishlens:smoke
NAME=phishlens-smoke
PORT=18000
URL="http://127.0.0.1:$PORT"

cleanup() {
    docker rm --force "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build --tag "$IMAGE" .

# Read-only root, no capabilities, and a non-root user, as it should run in practice.
docker run --detach --name "$NAME" --publish "127.0.0.1:$PORT:8000" \
    --read-only --cap-drop ALL --security-opt no-new-privileges "$IMAGE" >/dev/null

tries=0
until curl --silent --fail "$URL/health" >/dev/null; do
    tries=$((tries + 1))
    if [ "$tries" -ge 30 ]; then
        echo "FAIL: the service did not become healthy"
        docker logs "$NAME"
        exit 1
    fi
    sleep 1
done
echo "ok: /health answered after $tries retries"

check_status() {
    expected=$1
    path=$2
    body=$3
    actual=$(curl --silent --output /dev/null --write-out '%{http_code}' \
        --header 'content-type: application/json' --data "$body" "$URL$path")
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: POST $path gave $actual, want $expected"
        exit 1
    fi
    echo "ok: POST $path gave $actual"
}

check_status 200 /v1/score '{"url": "https://example.org/about/team"}'
check_status 200 /v1/score/batch '{"urls": ["https://example.org/", "http://192.0.2.1/login"]}'
check_status 422 /v1/score '{"url": "javascript:alert(1)"}'
check_status 422 /v1/score/batch '{"urls": []}'

curl --silent --fail --header 'content-type: application/json' \
    --data '{"url": "http://paypal.secure-login.example.com/verify/account.php", "top": 3}' \
    "$URL/v1/score"
echo

user=$(docker exec "$NAME" id -u)
if [ "$user" = "0" ]; then
    echo "FAIL: the container runs as root"
    exit 1
fi
echo "ok: runs as uid $user"

if docker exec "$NAME" python -c "import pandas" 2>/dev/null; then
    echo "FAIL: the training stack is in the image"
    exit 1
fi
echo "ok: no training stack in the image"

# With no network at all, scoring must still work.
docker run --rm --network none "$IMAGE" phishlens score https://example.org/ --top 1
echo "ok: scored with no network"

docker image inspect "$IMAGE" --format 'image size: {{.Size}} bytes'
echo "smoke test passed"
