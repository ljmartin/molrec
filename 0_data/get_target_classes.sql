.headers on
.mode csv
.output target_classes.csv


SELECT target_dictionary.chembl_id, protein_family_classification.l1, protein_family_classification.l2, protein_family_classification.l3, target_dictionary.pref_name
FROM target_dictionary INNER Join target_components
          ON target_dictionary.tid = target_components.tid
INNER Join component_class
          ON target_components.component_id = component_class.component_id
INNER Join protein_family_classification
      ON component_class.protein_class_id = protein_family_classification.protein_class_id
WHERE target_dictionary.organism in ("Homo sapiens")
and target_dictionary.target_type in ("SINGLE PROTEIN");
