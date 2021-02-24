import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def convert(input_column,output_column, value,df):
    df[output_column]=np.nan
    for i in range(len(df[input_column])):
        if value in str(df[input_column][i]):
            df._set_value(i,output_column,int(1))
        else:df._set_value(i,output_column,int(0))
    return df

def CARE_data():
    path_enrollment = "/home/michal/MYOR Dropbox1/R&D/Clinical Trials/CARE- Comprehending Atopic Risk Elements/Observational_Trial_Tracking/Poriya and Meir Enrollment tracking New5.10.2020.xlsx"
    Sheet_enrollment_Meir = "Meir Enrollment"
    Sheet_enrollment_Poriya = "Poriya Enrollment"
    Skiprows_enrollment = [0]

    df1=pd.read_excel(path_enrollment, Sheet_enrollment_Meir, skiprows=Skiprows_enrollment)
    df2=pd.read_excel(path_enrollment, Sheet_enrollment_Poriya, skiprows=Skiprows_enrollment)
    df=pd.concat([df1,df2],ignore_index=True)

    df=df.drop(columns=['Unnamed: 0','Autoimmiun desease in family','Mother took antibiotics during pregnency', 'Gestational diabetes',
     'Unnamed: 26', 'Smoking father', 'Subject ID','Allergy type', 'Initials', 'Enrollment day', 'Birth day','First bath','Fototherapy', 'Apgar Score', 'Unnamed: 12', 'Number of child ',
     'Mother came in touch with farm animals', 'Mother year of birthday', 'Leaving environment', 'Place of Leaving','Age of going to kindergarden','AD arm risk score', 'FA arm risk score', 'AD head risk score',
     'FA head risk score' , 'Hay Fever in family'])

    df = df.rename(columns={"Sex": "Child's sex", "Primary Ethnicity": "ethnicity", "Cesarean": "mode of delivery",
                            "Week of birth": "gestational age",
                            "Weight (in grams)": "birth weight kg",
                            "Smoking mother": "mother smoked during pregnancy",
                            "Three months brestfeading": "Exclusively Breastfed",
                            "Eczema (Mother)": "mother has eczema", "Eczema (Father)": "father has eczema",
                            "Asthma (Mother)": "mother has asthma",
                            "Asthma (Father)": "father has asthma", "Food Allergy (Mother)": "mother has food allergy",
                            "Food Allergy (Father)": "father has food allergy"})
    df["birth weight kg"]=df["birth weight kg"]/1000
    df["any pets owned at enrolment"] = np.logical_or(df["Cat in the family"], df["Dog in the family"])
    df["any pets owned at enrolment"] = df["any pets owned at enrolment"].map({True:int(1), False:int(0)})
    df=df.drop(columns=["Cat in the family","Dog in the family"])

    df=convert("AD in family","mother has eczema",'1',df)
    df=convert("AD in family","father has eczema",'2',df)
    df=convert("Asthma in family","father has asthma",'2',df)
    df=convert("Asthma in family","mother has asthma",'1',df)
    df=convert("Allergy in family","father has food allergy",'2',df)
    df=convert("Allergy in family","mother has food allergy",'1',df)
    df=df.drop(columns=["AD in family","Asthma in family","Allergy in family"])


    df["ethnicity_0"] = int(0)
    df["ethnicity_1"] = int(1)
    df["ethnicity_2"] = int(0)
    df["ethnicity_3"] = int(0)
    df["ethnicity_4"] = int(0)
    df=df.dropna(subset=["Child's sex", 'gestational age','mode of delivery','birth weight kg','Exclusively Breastfed','mother smoked during pregnancy','mother has eczema','mother has food allergy','mother has asthma','father has eczema','father has food allergy','father has asthma','any pets owned at enrolment'],thresh=3)
    # df = df[["Child's sex", "birth weight kg", "gestational age","mother has eczema","father has eczema","mother has asthma","father has asthma","mother has food allergy","father has food allergy","mode of delivery","any pets owned at enrolment",	"Exclusively Breastfed","mother smoked during pregnancy","ethnicity_0","ethnicity_1","ethnicity_2","ethnicity_3","ethnicity_4"]]
    return df

def EAT_Table(path, late_intro):
    print('EAT')
    # sex- 0=male
    # mode of delivery- 0=vaginal

    df = pd.read_excel(path)
    df = df.drop(columns=["Participant ID", "Dataset", "Season of Birth", "Farm Animals", "Urban/Rural",
                          "Vitamin D",
                          "mother smokes at enrolment", "father smokes at enrolment",
                          "wheat allergy at 12 months (defined by DBPCFC)",
                          "milk allergy at 12 months (defind by DBPCFC)",
                          "cod allergy at 12 months (defined by DBPCFC)",
                          "egg allergy at 12 months (defined by DBPCFC)",
                          "primary outcome sesame allergy (only those evaluable and within age range)",
                          "primary outcome fish allergy (only those evaluable and within age range)",
                          "primary outcome wheat allergy (only those evaluable and within age range)","study group",
                          "Moisturising cream - age at onset at enrolment (weeks)",
                          "Parent reports child had eczema before enrolment (parent and/or Dr diagnosed)"])

    df["SCORAD"]=df[["SCORAD at 3 month clinic visit", "SCORAD at 12 month clinic visit"]].max(axis=1)
    df["Exclusively Breastfed 3 months"]=int(1)
    df["Exclusively Breastfed 2 weeks"]=int(1)

    df=df.drop(columns=["Exclusively Breastfed","mother's age at enrolment (years)","mother has hayfever", "father has hayfever",'family history of eczema',"SCORAD at 12 month clinic visit","SCORAD at 3 month clinic visit",
       'family history of asthma', 'family history of hayfever', 'family history of food allergy','number of siblings at enrolment','attends childminder or nursery'])

    df=df.rename(columns={"primary outcome positive (only those evaluable and within age range)":"FA_general","primary outcome egg allergy (only those evaluable and within age range)":"FA_Egg","primary outcome milk allergy (only those evaluable and within age range)":"FA_Milk","primary outcome peanut allergy":"FA_Peanut"})

    res = pd.get_dummies(df["ethnicity"], prefix='ethnicity')
    df = df.drop(columns=["ethnicity"])
    df = pd.concat([df, res], axis=1)

    # print('general', Counter(df["FA_general"]))
    # print('MILK', Counter(df["FA_Milk"]))
    # print('EGG', Counter(df["FA_Egg"]))
    # print('PEANTUT', Counter(df["FA_Peanut"]))
    df["research"] =int(10)
    df["count"]=int(1)
    return df

def LEAP_db(path_LEAP,late_intro):
    # issues- how do they measure FA development?- BF + HN(allergic)
    # ethnicity- asian, white, black, chinese, mixed- 0/white, 1/mixed, 2/asian or asian british, 3/black or black british, 4/chinese or other ethnic group
    # breastfeeding- here in months for exsclusive- 3 months- binary
    # use only mother -father family history
    # what is the meaning here of vitamin D?- ריכוז בדם

    print('LEAP')
    df_AD=pd.read_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/LEAP/ECZ1MSTR_2021-02-23_00-05-02.xlsx",usecols="A:B,P")
    df_AD = df_AD.groupby('Participant ID').first()
    df_AD = df_AD[df_AD["Visit"] == 12]

    df1 = pd.read_excel(path_LEAP, sheet_name="raw_data", usecols="A,N,P:T,W:X,AB,AI:AL,AO:AR,BF,HN")
    df1=df1.dropna(subset=["Egg allergy by protocol definition"]) #remove unneccsary empty rows
    df = df1.merge(df_AD, on='Participant ID',how='outer')

    df["Exclusively Breastfed 3 months"] =np.where(df["Time Exclusively Breastfed (months)"] >= 3, int(1), int(0))
    df["Exclusively Breastfed 2 weeks"] =np.where(df["Time Exclusively Breastfed (months)"] >= 0.5, int(1), int(0))
    df['Exclusively Breastfed 2 weeks'] = df['Exclusively Breastfed 2 weeks'].fillna(0)
    df['Exclusively Breastfed 3 months'] = df['Exclusively Breastfed 3 months'].fillna(0)

    df=df.drop(columns=["Time Exclusively Breastfed (months)","Visit"])

    df["any pets owned at enrolment"] = np.logical_or(df["Cat(s) in Home?"], df["Dog(s) in Home?"])

    # produce allergy prediction coloumn
    df["Egg allergy by protocol definition"] = df["Egg allergy by protocol definition"].map({'Yes': True, 'No': False})
    df["Outcome of Peanut OFC at V60 (character)"] = df["Outcome of Peanut OFC at V60 (character)"].map(
        {'Allergic': True, 'Tolerant': False, 'Indeterminate': None, 0: None})
    df["FA_general"] = np.logical_or(
        df["Egg allergy by protocol definition"], df["Outcome of Peanut OFC at V60 (character)"].fillna(value=False))

    # drop the irrelevant tagging
    # if FA:
    #     df = df.drop(columns=["SCORAD"])
    # else:
    #     df = df.drop(columns=["primary outcome positive (only those evaluable and within age range)"])


    df = df.rename(columns={"Sex": "Child's sex", "Primary Ethnicity": "ethnicity", "Type of Birth": "mode of delivery",
                            "Gestational Age at Birth (weeks)": "gestational age",
                            "Birth Weight (kg)": "birth weight kg",
                            "Mother Smoke During Pregnancy?": "mother smoked during pregnancy",
                            "&gt;1 Day a Week Nursery/Daycare/Playgroup": "attends childminder or nursery",
                            "Eczema (Mother)": "mother has eczema", "Eczema (Father)": "father has eczema",
                            "Asthma (Mother)": "mother has asthma",
                            "Asthma (Father)": "father has asthma", "Food Allergy (Mother)": "mother has food allergy",
                            "Food Allergy (Father)": "father has food allergy","Egg allergy by protocol definition":"FA_Egg", "Outcome of Peanut OFC at V60 (character)":"FA_Peanut"})
    df = df.drop(columns=["Vitamin D", "Cat(s) in Home?", "Dog(s) in Home?", "Season of Birth", "Participant ID"])
    # df = df.dropna(axis=0)

    col = ["mother smoked during pregnancy", "mother has eczema", "father has eczema", "mother has asthma",
           "father has asthma", "mother has food allergy", "father has food allergy", "any pets owned at enrolment"]
    for i in col:
        df[i] = df[i].map({'Yes': int(1), 'No': int(0)})
    df["Child's sex"] = df["Child's sex"].map({1: int(0), 2: int(1)})
    df["mode of delivery"] = df["mode of delivery"].map({'Vaginal': int(0), "C-section": int(1)})
    #  0/white, 1/mixed, 2/asian or asian british, 3/black or black british, 4/chinese or other ethnic group

    df["ethnicity"] = df["ethnicity"].map(
        {"White": int(0), "Mixed": int(1), "Black": int(3), "Asian": int(2), "Chinese, Middle Eastern, or Other Ethnic Group": int(4),"Missing":int(1)})
    res = pd.get_dummies(df["ethnicity"], prefix='ethnicity')
    df = pd.concat([df, res], axis=1)
    df=df.rename(columns={"ethnicity_0.0": "ethnicity_0", "ethnicity_1.0": "ethnicity_1", "ethnicity_2.0": "ethnicity_2", "ethnicity_3.0": "ethnicity_3", "ethnicity_4.0": "ethnicity_4", })
    df = df.drop(columns=["ethnicity"])

    # df=df.dropna(subset=["Child's sex", 'gestational age','mode of delivery','birth weight kg','Exclusively Breastfed','mother smoked during pregnancy','mother has eczema','mother has food allergy','mother has asthma','father has eczema','father has food allergy','father has asthma','any pets owned at enrolment'])
    # df=df.dropna(subset=["FA_general"])

    df["research"] =int(20)
    df["count"]=int(1)

    # if FA:
    #     df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP_FA.xlsx",index=False)
    # else: df.to_excel("/home/michal/MYOR Dropbox/R&D/Partnerships/Tables/LEAP_AD.xlsx",index=False)
    # print('general', Counter(df["FA_general"]))
    # print('EGG', Counter(df["FA_Egg"]))
    # print('PEANTUT', Counter(df["FA_Peanut"]))
    print("LEAP table", df.shape)

    return df

def Katz_db(path_KATZ,late_intro):
    print('KATZ')
    # issues-all i nemail from Michael "Assaf Harofeh Database"

    df = pd.read_excel(path_KATZ, usecols="A,E,G:H,J:L,O,S,W,AB,AD:AF,AZ:BB,BU,BX,CC:CD")  # child history: CC:CE

    df['Exclusively Breastfed 2 weeks']=np.where(df["Exclusive BF To"]>=15,int(1),int(0))
    df['Exclusively Breastfed 3 months']=np.where(df["Exclusive BF To"]>=90,int(1),int(0))
    df['Exclusively Breastfed 2 weeks'] = df['Exclusively Breastfed 2 weeks'].fillna(0)
    df['Exclusively Breastfed 3 months'] = df['Exclusively Breastfed 3 months'].fillna(0)

    df = df.drop(columns=["Season", "Maternal Age", "Residance", "Newborn Order","Exclusive BF To"])
    df["Weight"] = df["Weight"] / 1000

    df = df.rename(columns={"Gender": "Child's sex", "Delivery Type": "mode of delivery",
                            "Week Gestation": "gestational age",
                            "Weight": "birth weight kg",
                            "maternal smoking": "mother smoked during pregnancy",
                            "maternal AD": "mother has eczema", "paternal AD": "father has eczema",
                            "maternal asthma": "mother has asthma", "child AD": "SCORAD",
                            "paternal asthma": "father has asthma", "maternal food allergy": "mother has food allergy",
                            "paternal food allergy": "father has food allergy",
                            "Diagnosis": "FA_general"})

    # df = df.dropna()
    df["mother has asthma"]=df["mother has asthma"].map({3:int(1),6:int(1),7:int(1),8:int(1),0:int(0),1:int(1)})
    df["mother has eczema"]=df["mother has eczema"].map({4:int(1),0:int(0),1:int(1)})
    df["mother has food allergy"]=df["mother has food allergy"].map({6:int(1),0:int(0),1:int(1)})
    df["father has food allergy"]=df["father has food allergy"].map({6:int(1),0:int(0),1:int(1)})
    df["father has asthma"]=df["father has asthma"].map({3:int(1),6:int(1),11:int(1),13:int(1),0:int(0),1:int(1)})
    df["any pets owned at enrolment"] = np.logical_or(df["pets at yard"], df["pets at home"])
    df = df.drop(columns=["pets at yard", "pets at home"])

    df["ethnicity_0"] = int(0)
    df["ethnicity_1"] = int(0)
    df["ethnicity_2"] = int(0)
    df["ethnicity_3"] = int(0)
    df["ethnicity_4"] = int(1)

    df["child food allergy"]=np.where(df["child food allergy"]>0,int(1),int(0))
    df["FA_general"]=df["FA_general"].map({1:int(1),2:int(1),3:int(0),4:int(0),5:int(0),6:int(1)})
    df["FA_general"]=np.logical_or(df["FA_general"],df["child food allergy"])
    df=df.drop(columns=["child food allergy"])

    df["FA_Milk"]=df["FA_general"]

    df["research"] = int(30)
    df["count"]=int(1)
    # print('general', Counter(df["FA_general"]))
    # print('MILK', Counter(df["FA_Milk"]))
    return df

def cofar2_Table(path_cofar,late_intro):
    print('CoFar')
    df1 = pd.read_excel(path_cofar, sheet_name="Attributes1", usecols="A:D,F:H,J:N,Q,V,X,AA,AB,AD")
    df2 = pd.read_excel(path_cofar, sheet_name="Attributes_to_be_Condensed", usecols="A,C,D")
    df2=df2.groupby('Accession').first()
    df=pd.merge(df1, df2, on='Accession', left_index=True)

    df['Exclusively Breastfed 2 weeks']=np.where(df["BRSTFEXB"]>1,int(1),int(0))
    df['Exclusively Breastfed 3 months']=np.where(df["BRSTFEXB"]>=3,int(1),int(0))

    df["PREGTERM"]=df["PREGTERM"].map({0: int(35),1:int(38)})
    df["PREGWEEK"]=df["PREGWEEK"].fillna(df["PREGTERM"])
    df["birth weight kg"]=0.453592*df["BIRTHLBS"]+0.0283495*df["BIRTHOZS"]
    df["RACE"]=df["RACE"].map({1: int(0),2:int(3),3: int(2),4:int(4),5:int(4),99:int(1)})
    df["SEX"]=df["SEX"].map({1: int(0),2:int(1)})
    df["FA_Egg"]=df["IN1AEVDN"].map({0:int(0),1: int(0),2:int(1),3:int(1)})
    df["FA_Milk"]=df["IN1AEVDN"].map({0:int(0),1: int(1),2:int(0),3:int(1)})

    df = df.rename(columns={"MATASTM": "mother has asthma","MATATD": "mother has eczema", "MATFOOD": "mother has food allergy",
                            "PATASTM": "father has asthma", "PATATD": "father has eczema","PATFOOD": "father has food allergy",
                            "PREGWEEK": "gestational age", "RACE":"ethnicity",
                            "CAESARIA":"mode of delivery",
                            "SEX": "Child's sex", "FURPETSV":"any pets owned at enrolment","PNTCONFI":"FA_Peanut",
                            "SMOKERSV": "mother smoked during pregnancy","ATDMODSV":"SCORAD"
                            })
    # df=df.dropna(subset=["FA_Egg","FA_Milk","FA_Peanut"],thresh=3)
    df["SCORAD"]=df["SCORAD"].map({0:int(0),1:int(20),2:int(20),3:int(20)})
    df["mother has asthma"]=df["mother has asthma"].map({9:None,0:int(0),1:int(1)})

    df["mother has eczema"]=df["mother has eczema"].map({9:None,0:int(0),1:int(1)})
    df["mother has food allergy"]=df["mother has food allergy"].map({9:None,0:int(0),1:int(1)})
    df["father has asthma"]=df["father has asthma"].map({9:None,0:int(0),1:int(1)})
    df["father has eczema"]=df["father has eczema"].map({9:None,0:int(0),1:int(1)})
    df["father has food allergy"]=df["father has food allergy"].map({9:None,0:int(0),1:int(1)})

    df["FA_general"]=df["FA_Egg"]+ df["FA_Milk"]+ df["FA_Peanut"]
    df["FA_general"]= np.where(df["FA_general"]>0,1,0)#.map({True:int(1), False:int(0)})

    res = pd.get_dummies(df["ethnicity"], prefix='ethnicity')
    df = pd.concat([df, res], axis=1)
    df=df.rename(columns={"ethnicity_0.0": "ethnicity_0","ethnicity_1.0": "ethnicity_1","ethnicity_2.0": "ethnicity_2","ethnicity_3.0": "ethnicity_3","ethnicity_4.0": "ethnicity_4"})
    df["ethnicity_4"] = int(0)

    df=df.drop(columns=["PREGTERM","IN1AEVDN","ethnicity","BIRTHOZS","BIRTHLBS","Accession","BRSTFEXB"])

    df["research"] = int(40)
    df["count"]=int(1)

    print("CoFar Study",df.shape)

    # print('general', Counter(df["FA_general"]))
    # print('MILK', Counter(df["FA_Milk"]))
    # print('EGG', Counter(df["FA_Egg"]))
    # print('PEANTUT', Counter(df["FA_Peanut"]))
    return df

from sklearn.neighbors import KNeighborsRegressor

def impute_model_basic(df):
  cols_nan = df.columns[df.isna().any()].tolist()
  cols_no_nan = df.columns.difference(cols_nan).values
  for col in cols_nan:
      test_data = df[df[col].isna()]
      train_data = df.dropna()
      knr = KNeighborsRegressor(n_neighbors=5).fit(train_data[cols_no_nan], train_data[col])
      df.loc[df[col].isna(), col] = knr.predict(test_data[cols_no_nan])
  return df

if __name__ == '__main__':
    path_EAT = "/home/michal/MYOR Dropbox/R&D/Partnerships/EAT/EAT_Risk_Score.xlsx"
    path_LEAP = "/home/michal/MYOR Dropbox/R&D/Partnerships/LEAP/LEAP_Data.xlsx"
    path_KATZ = "/home/michal/MYOR Dropbox/R&D/Partnerships/Katz_Study/Output dat_myor_milk_Oct    4_2020.xlsx"
    path_cofar2="/home/michal/MYOR Dropbox/R&D/Allergies Product Development/Prediction/Algorithm_Beta/Datasets/COFAR2/COFaR2_for_Risk_Score.xlsx"

    late_intro=False
    df4=cofar2_Table(path_cofar2, late_intro)
    df1=Katz_db(path_KATZ, late_intro)
    df2=LEAP_db(path_LEAP, late_intro)
    df3=EAT_Table(path_EAT, late_intro)
    DF = pd.concat([df1, df2, df3,df4])
    DF=DF.drop(columns=["FA_Egg","FA_Milk","FA_Peanut","SCORAD"])
    DF=DF.dropna(subset=["FA_general"])
    # DF=DF.dropna(thresh=16)
    # df["mother has eczema"] = df["mother has eczema"].map({4:int(1), 9:int(1)})
    print("DF",DF.shape)
    # print(Counter(DF["FA_general"]))

    # # DF["multi"]= np.logical_and(np.where(DF["SCORAD"].values > 0, 1, 0),np.where(DF["primary outcome positive (only those evaluable and within age range)"].values > 0, 1, 0))
    # # DF["multi"].replace(True, int(1), inplace=True)
    # # DF["multi"].replace(False, int(0), inplace=True)
    # # print("FA", Counter(np.where(DF["multi"] > 0, 1, 0)))
    # # DF=DF.drop(columns=["primary outcome positive (only those evaluable and within age range)","SCORAD"])
    DF.to_excel(f"./2302_no_drop_FA.xlsx",index=False)
    drop_thresh = 18
    df = DF.dropna(thresh=drop_thresh)
    df = impute_model_basic(df)
    df.to_excel(f"./2302_drop_FA.xlsx", index=False)
