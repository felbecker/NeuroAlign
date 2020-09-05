#renames all fasta files in the directory to uniformly named files A0.fasta, A1.fasta, A2.fasta...	
for i in *.fa; do
  mv -f "$i" "_$i"
done
a=0
for i in *.fa; do
  new=$(printf "A%04d.fasta" "$a") #04 pad to length of 4
  mv -f "$i" "$new"
  let a=a+1
done
