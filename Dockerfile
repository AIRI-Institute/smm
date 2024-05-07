FROM python:3.9
WORKDIR /code
COPY . .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
ENTRYPOINT ["tail", "-f", "/dev/null"]