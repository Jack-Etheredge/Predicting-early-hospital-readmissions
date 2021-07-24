"""Extracting the differences between the duplicated notebooks"""

if args.get(remove_expired):
    # Removing expired patients:
    patientdata = patientdata[patientdata.discharge_disposition_id.str.contains("Expired") == False]
    print(patientdata.shape)

if args.get(remove_duplicates):
    # Removing repeat patient entries (since they violate independence):
    patientdata = patientdata.groupby('patient_nbr', group_keys=False).apply(lambda x: x.loc[x.encounter_id.idxmin()])
    print(patientdata.shape)

if args.get(binary_classification):
    y = y.str.replace('>30','NO')
    y_test = y_test.str.replace('>30','NO')
    y_train = y_train.str.replace('>30','NO')