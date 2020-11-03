.headers on
.mode csv
.output interaction_data_ALL_species.csv


SELECT target_dictionary.chembl_id, target_dictionary.pref_name, molecule_dictionary.chembl_id, compound_structures.canonical_smiles
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
WHERE  (
		activities.pchembl_value > "5"
/*		or activities.activity_comment in ('Active', 'active', 'Partial agonist', 'Agonist', 'Antagonist')*/
	)
	and target_dictionary.organism in ('Homo sapiens', 'Mus musculus', 'Rattus norvegicus')
