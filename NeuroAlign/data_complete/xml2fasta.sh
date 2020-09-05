#converts all xml files in the directory to fasta files with the same name
for i in *.xml; do
  python3 xml2fasta.py -f="$i"
done
