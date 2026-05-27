from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["tutorial"]
)
def hello_pipeline():

    @task
    def say_hello():
        print("Hello from Airflow!")
        return "hello"

    @task
    def say_name(greeting: str):
        print(f"{greeting} — this is Varnit's ML pipeline")
        return "done"

    @task
    def say_done(status: str):
        print(f"Pipeline status: {status}")

    # This is how you chain tasks — output of one feeds into next
    greeting = say_hello()
    status = say_name(greeting)
    say_done(status)

hello_pipeline()