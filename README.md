# KSC-Microbiome-2023

The codes in this repository correspond to the analysis conducted in the study "Facial Skin Biophysical Multi-Parameter and Microbiome-Based Korean Skin Cutoype (KSC) Determination" by Mun et al. (2023) and are organized according to the description provided in the Methods section of the paper.


## Pairwise comparison tests of 12 groups (3 age groups x 4 KSC types) for 15 genera

```
# Prepare comma-separated samples per group which are used as an argument
for group in $(cat group.txt); do
	awk -F'\t' -vgroup=$group '($2 == group)' sample.group.txt | cut -f1 | tr '\n' ',' | sed 's/,$//' > $group.comma_separated_samples.txt
done

# Comparison tests using ANCOM-BC2 with 0.05 p-value cutoff and 8 CPU threads
for group1 in $(cat group.txt); do for group2 in $(cat group.txt | awk -vgroup1=$group1 '($o < group1)'); do
	Rscript ancombc.R DivCom_15genus_abundance_678ea.txt $group1.$group2.ancombc.tsv 0.05 8 $group1.comma_separated_samples.txt $group2.comma_separated_samples.txt
done; done

# Summarize comparison test outputs
for group1 in $(cat group.txt); do for group2 in $(cat group.txt | awk -vgroup1=$group1 '($o < group1)'); do
	perl table.rearrangeColumns.pl -c $group1.$group2.ancombc.tsv feature p lfc \
	| awk -F'\t' -vOFS='\t' -vgroup1=$group1 -vgroup2=$group2 '(NR > 1) {print $1, group1, group2, $2, $3}'
done; done > combination_ancombc.txt

# Generate p-value heatmap plots for 15 genera
for feature in $(cut -f1 DivCom_15genus_abundance_678ea.txt | awk '(NR > 1)'); do
	awk -F'\t' -vOFS='\t' -vfeature=$feature '($1 == feature) {print $2, $3, ($5 > 0 ? -log($4) : log($4)) / log(10)}' combination_ancombc.txt \
	| awk -F'\t' -vOFS='\t' '{print $1, $2, $3; print $2, $1, -$3;}' \
	| perl table.addColumns.pl group.txt 0 - 0 1,2 \
	| perl table.mergeLines.pl -s - 0 1 2 `cat group.txt` \
	| perl table.substitute_value.pl - '' 0 \
	| bash -c "cat <(cat group.txt | tr '\n' '\t' | sed 's/\t$/\n/' | sed 's/^/\t/') -" \
	| Rscript heatmap.R stdin $feature.heatmap.log10_p.pdf no_cluster_rows no_cluster_cols cellwidth=20 cellheight=20
done
```


## Building models to classify groups using 15 genera as features

```
# Create group directories
for group in $(cat group.txt); do
	mkdir $group
done

# Prepare tables of samples and their classes for training
for prefix in $(cut -f2 age_group.prefix.txt); do for ksc_group in HH HL LH LL; do
	awk -F'\t' -vOFS='\t' -vprefix=$prefix -vgroup=$prefix$ksc_group '(substr($2, 1, 1) == prefix) {print $1, ($2 == group ? 1 : 0)}' sample.group.txt > $prefix$ksc_group/sample.class.txt
done; done

# Prepare feature tables for training
for group in $(cat group.txt); do
	perl table.search.pl DivCom_15genus_abundance_678ea.txt 0 genus.txt 0 \
	| perl table.transpose.pl - \
	| perl table.addColumns.pl - 0 $group/sample.class.txt 0 1 \
	| sed '1 s/\t$/\tclass/' \
	| sed '1 s/^Taxon\t/sample\t/' \
	| grep -v -P '\t$' > $group/table.txt
done

# Training with CatBoost, 16 CPU threads and 16GB RAM
for group in $(cat group.txt); do 
	python3 train.CatBoostClassifier.py --eval_metric AUC --thread_count 16 --used_ram_limit 16gb --n_trials 100000 --timeout 100000 --feature_importance_file train.feature_importance.txt table.txt class train.model > train.log
done

# Create a summary table of validation AUC values
for group in $(cat group.txt); do
	sed -n 's/^validation AUC //p' $group/train.log \
	| awk -vOFS='\t' -vgroup=$group '{print group, $o}'
done > validation_AUC.txt

# Create a summary table of validation AUC values and feature importances
for group in $(cat group.txt); do
	awk -vOFS='\t' -vgroup=$group '(NR > 1) {print group, $o}' $group/train.feature_importance.txt
done \
| table.mergeLines.pl - 1 0 2 `cat group.txt` \
| sed 's/^g__//' \
| bash -c "cat <(table.transpose.pl validation_AUC.txt | sed 's/^/\t/') -" \
| table.transpose.pl - \
| sed '1 s/^\t\t/sample\tvalidation AUC\t/' > validation_AUC.feature_importance.txt

# Generate a heatmap plot
cut -f1,3- validation_AUC.feature_importance.txt \
| Rscript heatmap.R stdin feature_importance.pdf scale no_cluster_rows no_cluster_cols cellwidth=20 cellheight=20
```
