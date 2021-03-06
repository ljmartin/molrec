.headers on
.mode csv
.output uniprot_accessions.csv


SELECT target_dictionary.chembl_id, component_sequences.accession, target_dictionary.pref_name
FROM target_dictionary INNER Join target_components
          ON target_dictionary.tid = target_components.tid
INNER Join component_class
          ON target_components.component_id = component_class.component_id
INNER Join component_sequences
      ON component_class.component_id = component_sequences.component_id
WHERE target_dictionary.organism in ("Homo sapiens")

