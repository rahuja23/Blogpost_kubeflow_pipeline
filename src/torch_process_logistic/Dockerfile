FROM python:3.8-slim
WORKDIR /home/user
COPY requirements.txt /home/user
COPY .env /home/user
RUN pip install -r requirements.txt
COPY main.py /home/user
ENTRYPOINT ["python", "main.py"]