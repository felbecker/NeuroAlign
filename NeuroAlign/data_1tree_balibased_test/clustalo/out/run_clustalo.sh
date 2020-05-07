for i in `seq 1 1000`
do
  clustalo --infile=$(printf "../A%04d.fa" "$i") --seqtype=PROTEIN --outfile=$(printf "A%04d.fa" "$i") --outfmt=fa --threads=4 -v
done