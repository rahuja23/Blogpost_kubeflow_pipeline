<<<<<<< HEAD
FROM python:3.8-slim
WORKDIR /home/user
COPY requirements.txt /home/user
COPY .env /home/user
=======
from python:3.8-slim
WORKDIR /home/user
COPY requirements.txt /home/user
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
RUN pip install -r requirements.txt
COPY main.py /home/user
ENTRYPOINT ["python", "main.py"]