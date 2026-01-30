all: | htslib bcftools samtools eigensoft beagle plink polegon SLiM

SLiM:
	wget https://github.com/MesserLab/SLiM/releases/download/v5.1/SLiM.zip
	unzip SLiM.zip -d SLiM
	cd SLiM && mkdir build && cd build && cmake ../SLiM && make slim

polegon:
	git clone https://github.com/YunDeng98/POLEGON

htslib:
	wget https://github.com/samtools/htslib/releases/download/1.22.1/htslib-1.22.1.tar.bz2
	tar -jxf htslib-1.22.1.tar.bz2
	ln -s htslib-1.22.1 htslib
	# bz2 compression unneeded, and the library is not automatically provided
	cd htslib-1.22.1 && ./configure prefix=$(PWD) --disable-bz2 && make && make install

bcftools: htslib
	wget https://github.com/samtools/bcftools/releases/download/1.22/bcftools-1.22.tar.bz2
	tar -jxf bcftools-1.22.tar.bz2
	ln -s bcftools-1.22 bcftools
	# htslib-1.22 doesn't always compile because of libcurl not being available
	rm -rf bcftools-1.22/htslib-1.22
	ln -s htslib-1.22.1 bcftools-1.22/htslib-1.22.1
	# Install liftover plugin: see 
	wget -P bcftools-1.22/plugins http://raw.githubusercontent.com/freeseek/score/master/{score.{c,h},liftover.c}
	cd bcftools-1.22 && ./configure prefix=$(PWD) && make && make install

samtools:
	wget https://github.com/samtools/samtools/releases/download/1.22/samtools-1.22.tar.bz2
	tar -jxf samtools-1.22.tar.bz2
	ln -s samtools-1.22 samtools
	rm -rf samtools-1.22/htslib-1.22
	ln -s htslib-1.22.1 samtools-1.22/htslib-1.22.1
	cd samtools-1.22 && ./configure prefix=$(PWD) --without-curses --disable-bz2 && make && make install
	
beagle:
	wget https://faculty.washington.edu/browning/beagle/beagle.18May20.d20.jar

plink:
	wget http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20201019.zip
	unzip plink_linux_x86_64_20201019.zip -d bin/

eigensoft/src/convertf: eigensoft/src/Makefile
	cd eigensoft/src && make convertf

eigensoft/src/Makefile:
	git clone https://github.com/DReichLab/EIG eigensoft

eigensoft: eigensoft/src/convertf

.PHONY: eigensoft

clean:
	rm -rf geva relate htslib* bcftools* samtools* eigensoft bin beagle POLEGON SLiM