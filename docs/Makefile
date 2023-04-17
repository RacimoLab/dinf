all: guide/diagram.svg
	./build.sh

guide/diagram.svg: guide/diagram.tex
	make -C guide

clean:
	rm -fr _build
	make -C guide clean
