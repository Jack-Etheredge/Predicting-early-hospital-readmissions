import pandas as pd
import patsy as patsy
import seaborn as sns
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preprocess_data(smote=True,scale=True):
    df = pd.read_csv('diabetic_data.csv')

    df.admission_type_id.replace(
    list(range(1,9)),['Emergency',
    'Urgent',
    'Elective',
    'Newborn',
    'Not Available',
    'NULL',
    'Trauma Center',
    'Not Mapped'], inplace=True)

    id_list = ['Discharged to home',
    'Discharged/transferred to another short term hospital',
    'Discharged/transferred to SNF',
    'Discharged/transferred to ICF',
    'Discharged/transferred to another type of inpatient care institution',
    'Discharged/transferred to home with home health service',
    'Left AMA',
    'Discharged/transferred to home under care of Home IV provider',
    'Admitted as an inpatient to this hospital',
    'Neonate discharged to another hospital for neonatal aftercare',
    'Expired',
    'Still patient or expected to return for outpatient services',
    'Hospice / home',
    'Hospice / medical facility',
    'Discharged/transferred within this institution to Medicare approved swing bed',
    'Discharged/transferred/referred another institution for outpatient services',
    'Discharged/transferred/referred to this institution for outpatient services',
    'NULL',
    'Expired at home. Medicaid only, hospice.',
    'Expired in a medical facility. Medicaid only, hospice.',
    'Expired, place unknown. Medicaid only, hospice.',
    'Discharged/transferred to another rehab fac including rehab units of a hospital .',
    'Discharged/transferred to a long term care hospital.',
    'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    'Not Mapped',
    'Unknown/Invalid',
    'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    'Discharged/transferred to a federal health care facility.',
    'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    'Discharged/transferred to a Critical Access Hospital (CAH).']

    df.discharge_disposition_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)

    id_list = ['Physician Referral',
    'Clinic Referral',
    'HMO Referral',
    'Transfer from a hospital',
    'Transfer from a Skilled Nursing Facility (SNF)',
    'Transfer from another health care facility',
    'Emergency Room',
    'Court/Law Enforcement',
    'Not Available',
    'Transfer from critial access hospital',
    'Normal Delivery',
    'Premature Delivery',
    'Sick Baby',
    'Extramural Birth',
    'Not Available',
    'NULL',
    'Transfer From Another Home Health Agency',
    'Readmission to Same Home Health Agency',
    'Not Mapped',
    'Unknown/Invalid',
    'Transfer from hospital inpt/same fac reslt in a sep claim',
    'Born inside this hospital',
    'Born outside this hospital',
    'Transfer from Ambulatory Surgery Center',
    'Transfer from Hospice']

    df.admission_source_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)
    df.admission_source_id.head()

    df = df[df.discharge_disposition_id.str.contains("Expired") == False]

    # #### ICD9 codes from here:
    #     http://www.icd9data.com/

    numeric_code_ranges = [(1,139),
    (140,239),
    (240,279),
    (280,289),
    (290,319),
    (320,389),
    (390,459),
    (460,519),
    (520,579),
    (580,629),
    (630,677),
    (680,709),
    (710,739),
    (740,759),  
    (760,779),  
    (780,799),  
    (800,999)]

    ICD9_diagnosis_groups = ['Infectious And Parasitic Diseases',
    'Neoplasms',
    'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders',
    'Diseases Of The Blood And Blood-Forming Organs',
    'Mental Disorders',
    'Diseases Of The Nervous System And Sense Organs',
    'Diseases Of The Circulatory System',
    'Diseases Of The Respiratory System',
    'Diseases Of The Digestive System',
    'Diseases Of The Genitourinary System',
    'Complications Of Pregnancy, Childbirth, And The Puerperium',
    'Diseases Of The Skin And Subcutaneous Tissue',
    'Diseases Of The Musculoskeletal System And Connective Tissue',
    'Congential Anomalies',
    'Certain Conditions Originating In The Perinatal Period',
    'Symptoms, Signs, And Ill-Defined Conditions',
    'Injury And Poisoning']

    codes = zip(numeric_code_ranges, ICD9_diagnosis_groups)
    codeSet = set(codes)

    for num_range, diagnosis in codeSet:
        #print(num_range)
        oldlist = range(num_range[0],num_range[1]+1)
        oldlist = [str(x) for x in oldlist]
        newlist = [diagnosis] * len(oldlist)
        for curr_col in ['diag_1', 'diag_2', 'diag_3']:
            df[curr_col].replace(oldlist, newlist, inplace=True)

    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        df[curr_col].replace(oldlist, newlist, inplace=True)
        df.loc[df[curr_col].str.contains('V'), curr_col] = 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'
        df.loc[df[curr_col].str.contains('E'), curr_col] = 'Supplementary Classification Of External Causes Of Injury And Poisoning'
        df.loc[df[curr_col].str.contains('250'), curr_col] = 'Diabetes mellitus'

    cat_cols = 

    df = df.drop(['readmitted','encounter_id','patient_nbr'],axis=1)

    # Replace age ranges with numerical values:
    age_id = {'[0-10)':0, 
            '[10-20)':10, 
            '[20-30)':20, 
            '[30-40)':30, 
            '[40-50)':40, 
            '[50-60)':50,
            '[60-70)':60, 
            '[70-80)':70, 
            '[80-90)':80, 
            '[90-100)':90}
    df['age_group'] = df.age.replace(age_id)
    df = df.drop(['age'],axis=1)

    y, x = patsy.dmatrices(f'readmitted ~ {patsy_x_string}', data=df, return_type="dataframe")
    df2 = df.select_dtypes(include = ['type_of_insterest'])
    df2 = df.select_dtypes(include = ['type_of_insterest'])
    df2 = df.select_dtypes(include = ['type_of_insterest'])

    df.select_dtypes(exclude=["number","bool_","object_"])

    print(X.head())


    # pd.get_dummies()

    # y = df.readmitted

    # numeric_columns = list(df.select_dtypes("int64").columns)
    # numeric_columns.remove('encounter_id')
    # numeric_columns.remove('patient_nbr')
    # numeric_columns

    # scaler = preprocessing.StandardScaler()

    # x_scaled = x.copy()
    # x_scaled[numeric_columns] = scaler.fit_transform(x_scaled[numeric_columns])

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    # x_train_scaled = x_train.copy()
    # x_test_scaled = x_test.copy()

    # # fit_transform on train
    # x_train_scaled[numeric_columns] = scaler.fit_transform(x_train_scaled[numeric_columns])

    # # transform on test
    # x_test_scaled[numeric_columns] = scaler.transform(x_test_scaled[numeric_columns])

    # return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()