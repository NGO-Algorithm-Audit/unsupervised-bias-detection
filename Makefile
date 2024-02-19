install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -v

format:
	black .

lint:
	ruff check .