all: kokkos_cpu.pdf kokkos_gpu.pdf

%.pdf: %.tex %_commands.md
	latexmk -pdfxe -shell-escape $<

clean:
	latexmk -c

cleanall:
	latexmk -C
	rm -rf _minted-*
