# Removing expired patients:
patientdata = patientdata[patientdata.discharge_disposition_id.str.contains("Expired") == False]
print(patientdata.shape)

# Removing repeat patient entries (since they violate independence):
patientdata = patientdata.groupby('patient_nbr', group_keys=False).apply(lambda x: x.loc[x.encounter_id.idxmin()])
print(patientdata.shape)