FROM python:3.12.12-slim-bookworm

WORKDIR /app/serve
COPY . /app/serve

RUN chmod +x start.sh && pip install -r requirements.txt && pip cache purge

EXPOSE 8400

CMD ["sh", "start.sh"]