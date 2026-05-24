FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y curl && apt-get clean

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /app
COPY src/ /app/src/
COPY airflow/dags/ /opt/airflow/dags/