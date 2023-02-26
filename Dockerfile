# syntax=docker/dockerfile:experimental
FROM python:3.7-slim

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget --no-proxy -O - -q localhost:8080
# HEALTHCHECK NONE

ENV http_proxy=http://www-proxy.fkie.fraunhofer.de:3128/
ENV https_proxy=http://www-proxy.fkie.fraunhofer.de:3128/
ENV no_proxy="auth,localhost,127.0.0.1,192.168.0.1/24"

RUN apt-get update -qq && apt-get install --no-install-recommends -y wget g++
ENV PROJECT_DIR /usr/src/web_app

RUN mkdir -p ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

COPY requirements.txt ${PROJECT_DIR}

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ${PROJECT_DIR}

RUN mkdir -p ${PROJECT_DIR}/data
EXPOSE 8080

WORKDIR ${PROJECT_DIR}/labelling_api
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python","-u", "survey_db_utils.py.py"]