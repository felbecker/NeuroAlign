for i in `seq 1 201`
do
  clustalo --infile=$(printf "../A%04d.fa" "$i") --seqtype=DNA --outfile=$(printf "A%04d.fa" "$i") --outfmt=fa --threads=4 -v
done
