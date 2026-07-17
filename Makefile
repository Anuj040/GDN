# Makefile
.PHONY: all-format

all-format:
	@echo "Formatting all Python files in directoy: $(DIR)"
	FILES=$$(find $(DIR) -name "*.py" -not -path "*/venv/*" | grep -v "__init__.py"); \
	uv run black --target-version py313 $$FILES; \
	uv run isort $$FILES; \
	uv run autoflake --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $$FILES; \
	echo "Formatting completed."