import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def EAT_Table(path, merged):
    # issues-
    # sex- 0=male
    # mode of delivery- 0=vaginal

    df = pd.read_excel(path)
    df = df.drop(columns=["Participant ID", "Dataset", "study group", "Season of Birth", "Farm Animals", "Urban/Rural",
                          "Vitamin D",
                          "mother smokes at enrolment", "father smokes at enrolment",
                          "wheat allergy at 12 months (defined by DBPCFC)",
                          "milk allergy at 12 months (defind by DBPCFC)",
                          "cod allergy at 12 months (defined by DBPCFC)",
                          "egg allergy at 12 months (defined by DBPCFC)",
                          "primary outcome egg allergy (only those evaluable and within age range)",
                          "primary outcome milk allergy (only those evaluable and within age range)",
                          "primary outcome sesame allergy (only those evaluable and within age range)",
                          "primary outcome fish allergy (only those evaluable and within age range)",
                          "primary outcome wheat allergy (only those evaluable and within age range)",
                          "Moisturising cream - age at onset at enrolment (weeks)",
                          "Parent reports child had eczema before enrolment (parent and/or Dr diagnosed)"])

    #drop for merging disserent tables
    if merged:
        df=df.drop(columns=["mother's age at enrolment (years)","mother has hayfever", "father has hayfever",'family history of eczema',
       'family history of asthma', 'family history of hayfever', 'family history of food allergy','number of siblings at enrolment','attends childminder or nursery'])

    df=df.rename(columns={"SCORAD at 12 month clinic visit":"SCORAD"})
    df=df.drop(columns=["primary outcome positive (only those evaluable and within age range)"])

    print(df.columns)
    df = df.dropna(axis=0)
    res = pd.get_dummies(df["ethnicity"], prefix='ethnicity')
    df = df.drop(columns=["ethnicity"])
    df = pd.concat([df, res], axis=1)
    print("EAT table",df.shape)

    if merged:
        df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT.xlsx",index=False)
    else: df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/EAT_only.xlsx",index=False)


def LEAP_db(path_LEAP):
    # issues- how do they measure FA development?- BF + HN(allergic)
    # ethnicity- asian, white, black, chinese, mixed- 0/white, 1/mixed, 2/asian or asian british, 3/black or black british, 4/chinese or other ethnic group
    # female=2
    # breastfeeding- here in months for exsclusive- 3 months- binary
    # use only mother -father family history
    # what is the meaning here of vitamin D?- ריכוז בדם

    df = pd.read_excel(path_LEAP, sheet_name="raw_data", usecols="A,N,P:T,W:X,AB,AI:AL,AO:AR,BZ,BF,HN")
    df["any pets owned at enrolment"] = np.logical_or(df["Cat(s) in Home?"], df["Dog(s) in Home?"])

    # produce allergy prediction coloumn
    df["Egg allergy by protocol definition"] = df["Egg allergy by protocol definition"].map({'Yes': True, 'No': False})
    df["Outcome of Peanut OFC at V60 (character)"] = df["Outcome of Peanut OFC at V60 (character)"].map(
        {'Allergic': True, 'Tolerant': False, 'Indeterminate': False, 0: False})
    df["primary outcome positive (only those evaluable and within age range)"] = np.logical_or(
        df["Egg allergy by protocol definition"], df["Outcome of Peanut OFC at V60 (character)"])

    df = df.drop(columns=["Egg allergy by protocol definition", "Outcome of Peanut OFC at V60 (character)"])

    # drop the irrelevant tagging
    df = df.drop(columns=["primary outcome positive (only those evaluable and within age range)"])

    df = df.rename(columns={"Sex": "Child's sex", "Primary Ethnicity": "ethnicity", "Type of Birth": "mode of delivery",
                            "Gestational Age at Birth (weeks)": "gestational age",
                            "Birth Weight (kg)": "birth weight kg",
                            "Mother Smoke During Pregnancy?": "mother smoked during pregnancy",
                            "&gt;1 Day a Week Nursery/Daycare/Playgroup": "attends childminder or nursery",
                            "Time Exclusively Breastfed (months)": "Exclusively Breastfed",
                            "Eczema (Mother)": "mother has eczema", "Eczema (Father)": "father has eczema",
                            "Asthma (Mother)": "mother has asthma",
                            "Asthma (Father)": "father has asthma", "Food Allergy (Mother)": "mother has food allergy",
                            "Food Allergy (Father)": "father has food allergy"})
    df = df.drop(columns=["Vitamin D", "Cat(s) in Home?", "Dog(s) in Home?", "Season of Birth", "Participant ID"])
    df = df.dropna(axis=0)

    col = ["mother smoked during pregnancy", "mother has eczema", "father has eczema", "mother has asthma",
           "father has asthma", "mother has food allergy", "father has food allergy", "any pets owned at enrolment"]
    for i in col:
        df[i] = df[i].map({'Yes': int(1), 'No': int(0)})
    df["Child's sex"] = df["Child's sex"].map({1: int(0), 2: int(1)})
    df["mode of delivery"] = df["mode of delivery"].map({'Vaginal': 0, "C-section": 1})
    #  0/white, 1/mixed, 2/asian or asian british, 3/black or black british, 4/chinese or other ethnic group

    df["ethnicity"] = df["ethnicity"].map(
        {"White": 0, "Mixed": 1, "Black": 3, "Asian": 2, "Chinese, Middle Eastern, or Other Ethnic Group": 4})
    res = pd.get_dummies(df["ethnicity"], prefix='ethnicity')
    df = pd.concat([df, res], axis=1)
    df.loc[df['Exclusively Breastfed'] <= 3, 'Exclusively Breastfed'] = int(0)
    df.loc[df['Exclusively Breastfed'] > 3, 'Exclusively Breastfed'] = int(1)

    df = df.drop(columns=["ethnicity"])

    print("LEAP table", df.shape)
    df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP.xlsx",index=False)


def Katz_db(path_KATZ):
    # issues-all i nemail from Michael "Assaf Harofeh Database"

    df = pd.read_excel(path_KATZ, usecols="A,E,G:H,J:L,O,S,W,AB,AD:AF,AZ:BB,BU,BX,CC")  # child history: CC:CE
    df = df.drop(columns=["Season", "Maternal Age", "Residance", "Newborn Order"])

    # drop the irrelevant tagging
    df = df.drop(columns=["Diagnosis"])

    df["Weight"] = df["Weight"] / 1000

    df = df.rename(columns={"Gender": "Child's sex", "Delivery Type": "mode of delivery",
                            "Week Gestation": "gestational age",
                            "Weight": "birth weight kg",
                            "maternal smoking": "mother smoked during pregnancy",
                            "Exclusive BF To": "Exclusively Breastfed",
                            "maternal AD": "mother has eczema", "paternal AD": "father has eczema",
                            "maternal asthma": "mother has asthma", "child AD": "SCORAD",
                            "paternal asthma": "father has asthma", "maternal food allergy": "mother has food allergy",
                            "paternal food allergy": "father has food allergy",
                            "Diagnosis": "primary outcome positive (only those evaluable and within age range)"})
    df.loc[df['Exclusively Breastfed'] <= 90, 'Exclusively Breastfed'] = int(0)
    df.loc[df['Exclusively Breastfed'] > 90, 'Exclusively Breastfed'] = int(1)

    df = df.dropna()

    df["any pets owned at enrolment"] = np.logical_or(df["pets at yard"], df["pets at home"])
    df = df.drop(columns=["pets at yard", "pets at home"])

    df["ethnicity_0"] = int(0)
    df["ethnicity_1"] = int(0)
    df["ethnicity_2"] = int(0)
    df["ethnicity_3"] = int(0)
    df["ethnicity_4"] = int(1)

    print("KATZ table", df.shape)
    df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/KATZ.xlsx",index=False)



