#!/usr/bin/env bash
# Развёртывание «Фенолога» на сервере.
#
#   ssh dmitry@188.187.214.35
#   git clone https://github.com/dimon0804/FENOLOG.git ~/fenolog
#   cd ~/fenolog && bash deploy/deploy.sh
#
# Скрипт идемпотентен: повторный запуск обновляет образ и перезапускает
# контейнер, не трогая накопленные данные — они лежат в именованном томе и
# переживают пересоздание.
set -euo pipefail

IMAGE=fenolog:prod
NAME=fenolog
PORT=8010                 # локальный порт; наружу отдаёт nginx
VOLUME=fenolog-state
DOMAIN=fenolog.clv-digital.tech

cd "$(dirname "$0")/.."

echo "==> Собираю образ"
docker build -t "$IMAGE" .

echo "==> Останавливаю прежний контейнер, если он есть"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> Запускаю"
# Переменные подобраны под публичный показ, а не под пакетный счёт:
#   YEARS=6        шесть сезонов истории — норма считается уверенно
#   MAX_SCENES     потолок сцен на поле, чтобы один разбор не занял полчаса
#   TASK_WORKERS=3 столько разборов одновременно; больше упрётся в сеть
#   SIBLINGS_BUDGET на публичном показе район чаще холодный, поэтому потолок
#                   времени на сбор соседей поднят против стандартных 150 с
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p 127.0.0.1:${PORT}:8000 \
    -v "${VOLUME}":/app/data \
    -e FENOLOG_YEARS=6 \
    -e FENOLOG_MAX_SCENES=400 \
    -e FENOLOG_TASK_WORKERS=3 \
    -e FENOLOG_NEIGHBOURS=1 \
    -e FENOLOG_SIBLINGS_BUDGET=300 \
    "$IMAGE"

echo "==> Жду, пока поднимется"
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "    сервис отвечает"
        break
    fi
    sleep 2
    if [ "$i" = 30 ]; then
        echo "    НЕ ПОДНЯЛСЯ. Логи:"
        docker logs --tail 40 "$NAME"
        exit 1
    fi
done

curl -fsS "http://127.0.0.1:${PORT}/health"; echo

cat <<EOS

Готово. Дальше, если nginx ещё не настроен:

  sudo cp deploy/nginx-fenolog.conf /etc/nginx/sites-available/fenolog
  sudo ln -sf /etc/nginx/sites-available/fenolog /etc/nginx/sites-enabled/
  sudo mkdir -p /var/www/fenolog
  sudo cp web/dist/_offline.html /var/www/fenolog/ 2>/dev/null || true
  sudo nginx -t && sudo systemctl reload nginx
  sudo certbot --nginx -d ${DOMAIN}

Проверить снаружи:  curl -I https://${DOMAIN}/health
EOS
