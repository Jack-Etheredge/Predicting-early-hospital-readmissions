"""Extracting the differences between the duplicated notebooks"""

import click

@click.command()
@click.option("--remove_expired", default=True, help="Remove expired patients (True or False)?")
@click.option("--remove_duplicates", default=True, help="Remove duplicate (returned) patients (True or False)?")
@click.option("--binary_classification", default=True, help="Reduce classes to binary <30 day vs >30 readmission by combining >30 and NO (True or False)?")


def preprocess():
    """Placeholder for preprocessing steps that distinguishes branch points between notebooks"""
    if remove_expired:
        # Removing expired patients:
        patientdata = patientdata[patientdata.discharge_disposition_id.str.contains("Expired") == False]
        print(patientdata.shape)

    if remove_duplicates:
        # Removing repeat patient entries (since they violate independence):
        patientdata = patientdata.groupby('patient_nbr', group_keys=False).apply(lambda x: x.loc[x.encounter_id.idxmin()])
        print(patientdata.shape)

    if binary_classification:
        y = y.str.replace('>30','NO')
        y_test = y_test.str.replace('>30','NO')
        y_train = y_train.str.replace('>30','NO')

if __name__ == '__main__':
    preprocess()
