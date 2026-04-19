FROM vila-mil-backend-base:local

WORKDIR /app

COPY deploy/requirements-api-app.txt /tmp/requirements-api-app.txt
RUN pip install -r /tmp/requirements-api-app.txt

COPY . /app

EXPOSE 8000

# gthread：同进程多线程并发，轻量请求（如 /api/health）不易被单个长耗时 API 完全堵死（仍勿同时压多个重计算请求）。
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--worker-class", "gthread", "--threads", "4", "--timeout", "1800", "--graceful-timeout", "120", "--log-level", "info", "api_server:app"]
