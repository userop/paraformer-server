FROM paraformer_serve:base

WORKDIR /app/serve
COPY . /app/serve

EXPOSE 8400

RUN apt install -y ffmpeg
RUN chmod +x start.sh
CMD ["sh", "start.sh"]