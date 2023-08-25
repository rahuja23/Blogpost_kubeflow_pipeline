FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet', download_dir='/usr/local/nltk_data')"]
ENV MODEL_NAME=Inference
ENV API_TYPE=REST
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0
EXPOSE 5000
EXPOSE 9000
RUN chown -R 8888 /app
CMD exec seldon-core-microservice $MODEL_NAME  --service-type $SERVICE_TYPE