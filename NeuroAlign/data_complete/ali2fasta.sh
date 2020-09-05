#converts all ali files in the directory to fasta files with the same name
for i in *.ali; do
  python3 ali2fasta.py -f="$i"
done
