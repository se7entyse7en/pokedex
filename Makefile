.PHONY: build
build:
	pydockenv create --name=pokedex --file=pydockenv.toml .

.PHONY: get-dataset
get-dataset:
	pydockenv run ./download_data.sh

.PHONY: run-notebook
run-notebook:
	pydockenv run -p 8888 jupyter notebook -- --allow-root --ip=0.0.0.0
