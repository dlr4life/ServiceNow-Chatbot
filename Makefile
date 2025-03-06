.PHONY: help install test lint run clean docker-build docker-run backup

# For development:
# make setup-dev  # Initial setup
# make run       # Run development server
# make test      # Run tests
# make lint      # Run linting

# For Docker deployment:
# make docker-build  # Build containers
# make docker-run    # Run the application

# For maintenance:
# make backup            # Backup database
# make security-check    # Run security audit
# make clean            # Clean temporary files

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make run        - Run development server"
	@echo "  make clean      - Clean up temporary files"
	@echo "  make backup     - Backup database"
	@echo "  make docker-build - Build Docker containers"
	@echo "  make docker-run   - Run Docker containers"

install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	pytest tests/ -v --cov=app

lint:
	flake8 app/
	black app/
	isort app/

run:
	flask run --debug

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +

docker-build:
	docker-compose build

docker-run:
	docker-compose up

backup:
	@echo "Creating database backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	cp instance/chatbot.db "backups/chatbot_$$timestamp.db"
	@echo "Backup created in backups/chatbot_$$timestamp.db"

setup-dev: install
	pre-commit install
	cp .env.example .env
	flask db upgrade
	@echo "Development environment setup complete"

migrate:
	flask db migrate
	flask db upgrade

shell:
	flask shell

routes:
	flask routes

security-check:
	bandit -r app/
	safety check

dependencies-update:
	pip-compile requirements.in
	pip-compile requirements-dev.in
	pip-sync requirements.txt requirements-dev.txt

docs:
	sphinx-build -b html docs/source docs/build 