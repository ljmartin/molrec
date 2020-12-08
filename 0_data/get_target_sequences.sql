.headers on
.mode csv
.output target_sequences.csv


SELECT target_dictionary.chembl_id, component_sequences.sequence, target_dictionary.pref_name
FROM target_dictionary INNER Join target_components
          ON target_dictionary.tid = target_components.tid
INNER Join component_class
          ON target_components.component_id = component_class.component_id
INNER Join component_sequences
      ON target_components.component_id = component_sequences.component_id
WHERE target_dictionary.organism in ("Homo sapiens")
and target_dictionary.target_type in ("SINGLE PROTEIN");
