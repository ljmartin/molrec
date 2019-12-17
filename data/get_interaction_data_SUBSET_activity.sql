.headers on
.mode csv
.output interaction_data_SUBSET_activity.csv

###Getting all three protein families for single protein, with the year for time split cross validation:
SELECT target_dictionary.chembl_id, target_dictionary.organism, target_dictionary.pref_name, molecule_dictionary.chembl_id, activities.pchembl_value, compound_structures.canonical_smiles, docs.year
FROM compound_structures INNER JOIN molecule_dictionary
ON compound_structures.molregno = molecule_dictionary.molregno
INNER Join activities
      ON molecule_dictionary.molregno = activities.molregno
INNER Join assays
      ON activities.assay_id = assays.assay_id
INNER Join docs
      ON assays.doc_id = docs.doc_id
INNER Join target_dictionary
      ON assays.tid = target_dictionary.tid
INNER Join target_components
          ON target_dictionary.tid = target_components.tid
INNER Join component_class
          ON target_components.component_id = component_class.component_id
INNER Join protein_family_classification
      ON component_class.protein_class_id = protein_family_classification.protein_class_id
WHERE target_dictionary.organism in ('Homo sapiens')
      and (activities.pchembl_value > "5"
                or activities.activity_comment in ('Active', 'active', 'Partial agonist', 'Agonist', 'Antagonist'))
		
	and target_dictionary.pref_name in ('11-beta-hydroxysteroid dehydrogenase 1', '3-phosphoinositide dependent protein kinase-1', '5-lipoxygenase activating protein', 'ADAM17', 'ALK tyrosine kinase receptor', 'Acetyl-CoA carboxylase 2', 'Acetylcholinesterase', 'Adenosine A1 receptor', 'Adenosine A2a receptor', 'Adenosine A2b receptor', 'Adenosine A3 receptor', 'Alpha-1a adrenergic receptor', 'Alpha-1b adrenergic receptor', 'Alpha-1d adrenergic receptor', 'Alpha-2a adrenergic receptor', 'Anandamide amidohydrolase', 'Androgen Receptor', 'Arachidonate 5-lipoxygenase', 'Beta secretase 2', 'Beta-1 adrenergic receptor', 'Beta-2 adrenergic receptor', 'Beta-3 adrenergic receptor', 'Beta-secretase 1', 'Bradykinin B1 receptor', 'Bromodomain-containing protein 4', 'Butyrylcholinesterase', 'C-C chemokine receptor type 1', 'C-C chemokine receptor type 2', 'C-C chemokine receptor type 3', 'C-C chemokine receptor type 5', 'C-X-C chemokine receptor type 3', 'C-X-C chemokine receptor type 7', 'Cannabinoid CB1 receptor', 'Cannabinoid CB2 receptor', 'Carbonic anhydrase I', 'Carbonic anhydrase II', 'Carbonic anhydrase IX', 'Carbonic anhydrase XII', 'Caspase-3', 'Cathepsin K', 'Cathepsin L', 'Cathepsin S', 'Cholecystokinin B receptor', 'Coagulation factor X', 'Complement factor D', 'Corticotropin releasing factor receptor 1', 'Cyclin-dependent kinase 1', 'Cyclin-dependent kinase 2', 'Cyclin-dependent kinase 2/cyclin A', 'Cyclin-dependent kinase 4', 'Cyclin-dependent kinase 4/cyclin D1', 'Cyclin-dependent kinase 5/CDK5 activator 1', 'Cyclin-dependent kinase 9', 'Cyclooxygenase-2', 'Cytochrome P450 11B1', 'Cytochrome P450 11B2', 'Cytochrome P450 19A1', 'Cytochrome P450 3A4', 'Delta opioid receptor', 'Diacylglycerol O-acyltransferase 1', 'Dihydrofolate reductase', 'Dipeptidyl peptidase I', 'Dipeptidyl peptidase IV', 'Dopamine D1 receptor', 'Dopamine D2 receptor', 'Dopamine D3 receptor', 'Dopamine D4 receptor', 'Dopamine transporter', 'Dual specificity mitogen-activated protein kinase kinase 1', 'Dual specificity protein kinase TTK', 'Endothelin receptor ET-A', 'Epidermal growth factor receptor', 'Epidermal growth factor receptor erbB1', 'Epoxide hydratase', 'Estrogen receptor alpha', 'Estrogen receptor beta', 'Fatty acid synthase', 'Fibroblast growth factor receptor 1', 'Fibroblast growth factor receptor 2', 'Fibroblast growth factor receptor 3', 'Fibroblast growth factor receptor 4', 'Focal adhesion kinase 1', 'Free fatty acid receptor 1', 'G protein-coupled receptor 44', 'GABA-A receptor; alpha-1/beta-3/gamma-2', 'GABA-A receptor; alpha-3/beta-3/gamma-2', 'GABA-A receptor; alpha-5/beta-3/gamma-2', 'Gamma-secretase', 'Ghrelin receptor', 'Glucagon receptor', 'Glucocorticoid receptor', 'Glucose-dependent insulinotropic receptor', 'Glycine transporter 1', 'Glycogen synthase kinase-3 beta', 'Gonadotropin-releasing hormone receptor', 'HERG', 'Heat shock protein HSP 90-alpha', 'Hepatocyte growth factor receptor', 'Hexokinase type IV', 'Histamine H1 receptor', 'Histamine H3 receptor', 'Histamine H4 receptor', 'Histone deacetylase', 'Histone deacetylase 1', 'Histone deacetylase 2', 'Histone deacetylase 3', 'Histone deacetylase 6', 'Histone deacetylase 8', 'Inhibitor of apoptosis protein 3', 'Inhibitor of nuclear factor kappa B kinase beta subunit', 'Insulin-like growth factor I receptor', 'Integrin alpha-4/beta-1', 'Integrin alpha-IIb/beta-3', 'Integrin alpha-V/beta-3', 'Interleukin-1 receptor-associated kinase 4', 'Interleukin-8 receptor B', 'Isocitrate dehydrogenase [NADP] cytoplasmic', 'Kappa opioid receptor', 'Kinesin-like protein 1', 'LXR-beta', 'Leucine-rich repeat serine/threonine-protein kinase 2', 'Leukocyte elastase', 'MAP kinase ERK2', 'MAP kinase p38 alpha', 'MAP kinase signal-integrating kinase 2', 'Macrophage colony stimulating factor receptor', 'Maternal embryonic leucine zipper kinase', 'Matrix metalloproteinase 13', 'Matrix metalloproteinase 3', 'Matrix metalloproteinase 8', 'Matrix metalloproteinase 9', 'Matrix metalloproteinase-1', 'Matrix metalloproteinase-2', 'Melanin-concentrating hormone receptor 1', 'Melanocortin receptor 4', 'Melatonin receptor 1A', 'Melatonin receptor 1B', 'Metabotropic glutamate receptor 1', 'Metabotropic glutamate receptor 2', 'Metabotropic glutamate receptor 5', 'Methionine aminopeptidase 2', 'Mineralocorticoid receptor', 'Mitogen-activated protein kinase kinase kinase 12', 'Mitogen-activated protein kinase kinase kinase 14', 'Monoamine oxidase B', 'Mu opioid receptor', 'Muscarinic acetylcholine receptor M1', 'Muscarinic acetylcholine receptor M2', 'Muscarinic acetylcholine receptor M3', 'Nerve growth factor receptor Trk-A', 'Neurokinin 1 receptor', 'Neurokinin 3 receptor', 'Neuronal acetylcholine receptor; alpha4/beta2', 'Neuropeptide Y receptor type 5', 'Nicotinamide phosphoribosyltransferase', 'Nociceptin receptor', 'Norepinephrine transporter', 'Nuclear receptor ROR-gamma', 'Orexin receptor 1', 'Orexin receptor 2', 'P2X purinoceptor 3', 'P2X purinoceptor 7', 'PI3-kinase p110-alpha subunit', 'PI3-kinase p110-alpha/p85-alpha', 'PI3-kinase p110-beta subunit', 'PI3-kinase p110-delta subunit', 'PI3-kinase p110-gamma subunit', 'Peroxisome proliferator-activated receptor alpha', 'Peroxisome proliferator-activated receptor delta', 'Peroxisome proliferator-activated receptor gamma', 'Phosphodiesterase 10A', 'Phosphodiesterase 4', 'Phosphodiesterase 4B', 'Phosphodiesterase 5A', 'Plasma kallikrein', 'Platelet-derived growth factor receptor beta', 'Poly [ADP-ribose] polymerase-1', 'Progesterone receptor', 'Prostaglandin E synthase', 'Prostanoid EP1 receptor', 'Protein farnesyltransferase', 'Protein kinase C alpha', 'Protein kinase C theta', 'Proteinase activated receptor 4', 'Purinergic receptor P2Y12', 'Receptor protein-tyrosine kinase erbB-2', 'Renin', 'Rho-associated protein kinase 1', 'Rho-associated protein kinase 2', 'Ribosomal protein S6 kinase 1', 'Serine/threonine-protein kinase AKT', 'Serine/threonine-protein kinase AKT2', 'Serine/threonine-protein kinase Aurora-A', 'Serine/threonine-protein kinase Aurora-B', 'Serine/threonine-protein kinase B-raf', 'Serine/threonine-protein kinase Chk1', 'Serine/threonine-protein kinase PIM1', 'Serine/threonine-protein kinase PIM2', 'Serine/threonine-protein kinase PIM3', 'Serine/threonine-protein kinase RAF', 'Serine/threonine-protein kinase mTOR', 'Serotonin 1a (5-HT1a) receptor', 'Serotonin 1b (5-HT1b) receptor', 'Serotonin 1d (5-HT1d) receptor', 'Serotonin 2a (5-HT2a) receptor', 'Serotonin 2b (5-HT2b) receptor', 'Serotonin 2c (5-HT2c) receptor', 'Serotonin 3a (5-HT3a) receptor', 'Serotonin 6 (5-HT6) receptor', 'Serotonin 7 (5-HT7) receptor', 'Serotonin transporter', 'Sigma opioid receptor', 'Smoothened homolog', 'Sodium channel protein type IX alpha subunit', 'Sodium/glucose cotransporter 1', 'Sodium/glucose cotransporter 2', 'Sphingosine 1-phosphate receptor Edg-1', 'Sphingosine 1-phosphate receptor Edg-3', 'Stem cell growth factor receptor', 'TGF-beta receptor type I', 'Thrombin', 'Thromboxane A2 receptor', 'Thromboxane-A synthase', 'Trypsin I', 'Tumour suppressor p53/oncoprotein Mdm2', 'Tyrosine-protein kinase ABL', 'Tyrosine-protein kinase BTK', 'Tyrosine-protein kinase ITK/TSK', 'Tyrosine-protein kinase JAK1', 'Tyrosine-protein kinase JAK2', 'Tyrosine-protein kinase JAK3', 'Tyrosine-protein kinase LCK', 'Tyrosine-protein kinase SRC', 'Tyrosine-protein kinase SYK', 'Tyrosine-protein kinase TIE-2', 'Tyrosine-protein kinase TYK2', 'Tyrosine-protein kinase receptor FLT3', 'Vanilloid receptor', 'Vascular endothelial growth factor receptor 1', 'Vascular endothelial growth factor receptor 2', 'c-Jun N-terminal kinase 1', 'c-Jun N-terminal kinase 3', 'p53-binding protein Mdm-2')
order by target_dictionary.tid