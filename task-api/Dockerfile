FROM python:3.9-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y curl wget gcc git build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app/

EXPOSE 80

HEALTHCHECK CMD curl --fail http://0.0.0.0/ping || exit 1

CMD ["uvicorn", "serve:api", "--host", "0.0.0.0", "--port", "80"]
