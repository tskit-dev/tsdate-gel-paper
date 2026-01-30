# Makefile for converting ancestry files to hg19-vcf and lifting over to GRCh38

# Define variables for file names and tools
TARGET = aadr_v62.0_1240K_public
CONVERTF = ../../tools/eigensoft/src/convertf
PLINK = ../../tools/bin/plink
BCFTOOLS = ../../tools/bin/bcftools
PAR_FILE = par.PACKEDANCESTRYMAP.PACKEDPED

# Reference and chain file variables
HG38_FA = hg38.fa
HG38_FA_GZ = $(HG38_FA).gz
CHAIN_FILE = hg19ToHg38.over.chain.gz
HG38_URL = https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/$(HG38_FA_GZ)

# Final target includes both hg19 and hg38 indexed VCF files
all: $(TARGET)_hg19.vcf.gz.tbi $(TARGET)_hg38.bcf $(TARGET).anno

# Index hg38 VCF
$(TARGET)_hg38.vcf.gz.tbi: $(TARGET)_hg38.vcf.gz
	../../tools/bin/tabix -p vcf $<

# Compress hg38 VCF
$(TARGET)_hg38.vcf.gz: $(TARGET)_hg38.vcf
	../../tools/bin/bgzip -c $< > $@

# Lift over from hg19 to hg38. We don't specify the fasta file for
# the input hg19 genome, because some of the refs in the input hg19 VCF
# are wrong, which means the liftover fails.
$(TARGET)_hg38.bcf: $(TARGET)_hg19.vcf.gz $(HG38_FA) $(CHAIN_FILE)
	$(BCFTOOLS) +liftover --no-version -Ou \
	  $< -- \
	  -f $(HG38_FA) \
	  -c $(CHAIN_FILE) \
	  --reject $(TARGET)_hg38.rejected_variants.bcf \
	  --reject-type b \
	  --write-src | \
	$(BCFTOOLS) sort -o $@ -Ob --write-index &> bcftools_liftover.log

# Download and prepare reference files
$(HG38_FA): $(HG38_FA_GZ)
	gunzip -c $< > $@

$(HG38_FA_GZ):
	curl -L -o $@ $(HG38_URL)

$(HG19_FA): $(HG19_FA_GZ)
	gunzip -c $< > $@

$(HG19_FA_GZ):
	curl -L -o $@ $(HG19_URL)

$(CHAIN_FILE):
	curl -L -o $@ https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/$@

# Index hg19 VCF
$(TARGET)_hg19.vcf.gz.tbi: $(TARGET)_hg19.vcf.gz 
	../../tools/bin/tabix -p vcf $<

# Convert to VCF using PLINK
$(TARGET)_hg19.vcf.gz: $(TARGET).bed $(TARGET).bim $(TARGET).fam
	$(PLINK) --keep-allele-order --output-chr chr26 --bfile $(TARGET) --recode vcf bgz --out $(TARGET)_hg19

# Create to bed/bim/fam files (Plink PackedPED format) using convertf
$(TARGET).bed $(TARGET).bim $(TARGET).fam: $(TARGET).geno $(TARGET).snp $(TARGET).ind $(PAR_FILE)
	$(CONVERTF) -p $(PAR_FILE)

# Extract the required files from the tar archive
$(TARGET).ind:
	curl -L -o $(TARGET).ind https://dataverse.harvard.edu/api/access/datafile/10537414

$(TARGET).snp:
	curl -L -o $(TARGET).snp https://dataverse.harvard.edu/api/access/datafile/10537415

$(TARGET).geno:
	curl -L -o $(TARGET).geno https://dataverse.harvard.edu/api/access/datafile/10537126

$(TARGET).anno:
	curl -L -o $(TARGET).anno https://dataverse.harvard.edu/api/access/datafile/10537413


# Create the parameter file
$(PAR_FILE):
	@echo "genotypename: $(TARGET).geno" > $(PAR_FILE)
	@echo "snpname: $(TARGET).snp" >> $(PAR_FILE)
	@echo "indivname: $(TARGET).ind" >> $(PAR_FILE)
	@echo "outputformat: PACKEDPED" >> $(PAR_FILE)
	@echo "genotypeoutname: $(TARGET).bed" >> $(PAR_FILE)
	@echo "snpoutname: $(TARGET).bim" >> $(PAR_FILE)
	@echo "indivoutname: $(TARGET).fam" >> $(PAR_FILE)

# Extract specific chromosome (usage: make chr CHROM=20)
.PHONY: chr
chr: $(TARGET)_hg38.bcf
	$(BCFTOOLS) view $< --regions chr$(CHROM) -O b -o $(TARGET)_hg38_chr$(CHROM).bcf

# Clean target to remove generated files
.PHONY: clean
clean:
	rm -f $(TARGET).bed $(TARGET).bim $(TARGET).fam $(PAR_FILE)
	rm -f $(TARGET)_hg19*
	rm -f $(TARGET)_hg38.bcf $(TARGET)_hg38.bcf.csi
	rm -f $(TARGET)_hg38.rejected_variants.bcf bcftools_liftover.log
	rm -f $(TARGET)_hg38_chr*.bcf


# Deep clean target to remove everything including downloaded files
.PHONY: distclean
distclean: clean
	rm -f $(TARGET).geno $(TARGET).snp $(TARGET).ind $(TARGET).anno $(HG38_FA) $(HG38_FA_GZ) $(CHAIN_FILE)

# Default target
.DEFAULT_GOAL := all