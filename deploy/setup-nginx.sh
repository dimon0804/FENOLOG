set -e
echo "== контейнер"
curl -fsS http://127.0.0.1:8010/health && echo " <- жив"

echo "== кладу конфигурацию nginx"
sudo tee /etc/nginx/sites-available/fenolog > /dev/null <<'CONF'
server {
    listen 80;
    listen [::]:80;
    server_name fenolog.clv-digital.tech;

    proxy_connect_timeout 30s;
    proxy_send_timeout    600s;
    proxy_read_timeout    600s;

    gzip on;
    gzip_types application/json application/javascript text/css text/plain image/svg+xml;
    gzip_min_length 1024;
    client_max_body_size 8m;

    location / {
        proxy_pass http://127.0.0.1:8010;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }

    error_page 502 503 504 /_offline.html;
    location = /_offline.html { root /var/www/fenolog; internal; }
}
CONF

sudo mkdir -p /var/www/fenolog
sudo docker cp fenolog:/app/src/api/error_pages/503.html /var/www/fenolog/_offline.html 2>/dev/null || echo "страницу для прокси не скопировал"
sudo ln -sf /etc/nginx/sites-available/fenolog /etc/nginx/sites-enabled/fenolog
sudo nginx -t 2>&1 | tail -2
sudo systemctl reload nginx && echo "nginx перезагружен"

echo "== проверка через nginx"
curl -s -o /dev/null -w "по http: %{http_code}\n" -H "Host: fenolog.clv-digital.tech" http://127.0.0.1/health

echo "== выпускаю сертификат"
sudo certbot --nginx -d fenolog.clv-digital.tech --non-interactive --agree-tos -m doviryak.dmitry@yandex.ru --redirect 2>&1 | tail -6
echo ГОТОВО
