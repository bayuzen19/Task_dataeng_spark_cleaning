include .env
export

CURRENT_DATE := $(shell date +'%Y-%m-%d')
JDBC_URL := "jdbc:postgresql://${DIBIMBING_DE_POSTGRES_HOST}/${DIBIMBING_DE_POSTGRES_DB}"
JDBC_PROPERTIES := '{"user": "${DIBIMBING_DE_POSTGRES_ACCOUNT}", "password": "${DIBIMBING_DE_POSTGRES_PASSWORD}", "driver": "${DIBIMBING_DE_SPARK_DRIVER}", "stringtype": "${DIBIMBING_DE_SPARK_STRINGTYPE}"}'

help:
	@echo "## postgres			- Run a Postgres container, including its inter-container network. \n"
	@echo "## spark			- Run a Spark cluster, rebuild the postgres container, then create the destination tables \n"
	@echo "## challenge-spark-a		- Build both postgres and spark images and containers,ccreate the destination tablesreate the destination tables, overwrite the destination table.\n"
	@echo "## jupyter			- Spinup jupyter notebook for testing and validation purposes.\n"
	@echo "## clean			- Cleanup all running containers related to the challenge."

all: network postgres spark jupyter

network:
	@docker network inspect dibimbing-dataeng-network >/dev/null 2>&1 || docker network create dibimbing-dataeng-network

postgres: postgres-create postgres-create-table postgres-ingest-csv

postgres-create:
	@docker-compose -f ./docker/docker-compose-postgres.yml --env-file .env up -d
	@echo '__________________________________________________________'
	@echo 'Postgres container created at port ${DIBIMBING_DE_POSTGRES_PORT}...'
	@echo '__________________________________________________________'
	@echo 'Postgres Docker Host	: ${DIBIMBING_DE_POSTGRES_HOST}' &&\
		echo 'Postgres Account	: ${DIBIMBING_DE_POSTGRES_ACCOUNT}' &&\
		echo 'Postgres password	: ${DIBIMBING_DE_POSTGRES_PASSWORD}' &&\
		echo 'Postgres Db		: ${DIBIMBING_DE_POSTGRES_DB}'
	@sleep 5
	@echo '==========================================================='

postgres-create-table:
	@echo '__________________________________________________________'
	@echo 'Creating tables...'
	@echo '_________________________________________'
	@docker exec -it ${DIBIMBING_DE_POSTGRES_CONTAINER_NAME} psql -U ${DIBIMBING_DE_POSTGRES_ACCOUNT} -d ${DIBIMBING_DE_POSTGRES_DB} -f challenge_sql/ddl-retail.sql
	@echo '==========================================================='

postgres-ingest-csv:
	@echo '__________________________________________________________'
	@echo 'Ingesting CSV...'
	@echo '_________________________________________'
	@docker exec -it ${DIBIMBING_DE_POSTGRES_CONTAINER_NAME} psql -U ${DIBIMBING_DE_POSTGRES_ACCOUNT} -d ${DIBIMBING_DE_POSTGRES_DB} -f challenge_sql/ingest-retail.sql
	@echo '==========================================================='

spark: spark-create

spark-create:
	@echo '__________________________________________________________'
	@echo 'Creating Spark Cluster ...'
	@echo '__________________________________________________________'
	@docker build -t dataeng-dibimbing/challenges_spark -f ./docker/spark.Dockerfile .
	@docker-compose -f ./docker/docker-compose-spark.yml --env-file .env up -d
	@echo '==========================================================='

challenge-spark:
	@docker exec ${DIBIMBING_DE_SPARK_MASTER_CONTAINER_NAME} \
		spark-submit \
		--master spark://${DIBIMBING_DE_SPARK_MASTER_HOST_NAME}:${DIBIMBING_DE_SPARK_MASTER_PORT} \
		/challenge_spark/submission-test.py \

clean:
	@bash ./helper/goodnight.sh

jupyter:
	@echo '__________________________________________________________'
	@echo 'Creating Jupyter Notebook Cluster at http://localhost:${DIBIMBING_DE_JUPYTER_PORT}...'
	@echo '__________________________________________________________'
	@docker build -t dataeng-dibimbing/jupyter -f ./docker/jupyter.Dockerfile .
	@docker-compose -f ./docker/docker-compose-jupyter.yml --env-file .env up -d
	@echo 'Created...'
	@echo 'Processing token...'
	@sleep 10
	@docker logs dibimbing-dataeng-jupyter 2>&1 | grep '\?token\=' -m 1 | cut -d '=' -f2
	@echo '==========================================================='
