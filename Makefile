OBJ = keccak.py

TEST = test.py

INSTALL = requirements.txt

default: $(OBJ)
	python3 $(OBJ)

install: $(INSTALL)
	python3 -m pip install -r $(INSTALL)

test: $(TEST)
	python3 $(TEST)