diagram.svg: diagram.tex
	latex $^
	dvisvgm --font-format=woff -b papersize diagram.dvi

clean:
	rm -f diagram.{dvi,svg,aux,log}
