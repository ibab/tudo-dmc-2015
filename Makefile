all: build/train_with_new_features.txt

build/train_with_new_features.txt: data/train.txt feature_generation.py | build
	python2 feature_generation.py

build:
	mkdir -p build

clean:
	rm -rf build
