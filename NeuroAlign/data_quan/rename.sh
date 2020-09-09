#renames all fasta files in the directory to uniformly named files A0.fasta, A1.fasta, A2.fasta...
for i in *.ref; do
  mv -f "$i" "_$i"
done
a=1
for i in *.ref; do
  new=$(printf "A%04d.fa" "$a") #04 pad to length of 4
  mv -f "$i" "$new"
  let a=a+1
done
