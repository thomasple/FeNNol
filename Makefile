all: doc

.PHONY: doc

doc:
	pdoc --docformat numpy -o ./docs -t ./pdoc_templates/ src/fennol/