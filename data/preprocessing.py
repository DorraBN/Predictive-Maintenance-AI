import pandas as pd
def load_datafaq():
   df=pd.read_csv("data/maintenance_knowledge.csv")
   print(df.head())
   print(df.isnull().sum())
   print(df.duplicated().sum())
   return df

def prepare_data(df):
    df=df.copy()
    cat=df.select_dtypes(include="object").columns.tolist()
    for col in cat:
      df[col]=df[col].str.lower()
      df[col]=df[col].str.replace(r'[^\w\s]','',regex=True)
   
    
    return df

def prepare_doc(df):
   documents=[]
   for _,row in df.iterrows():
      document=f"""
      category:{row["category"]}
      question:{row["question"]}
      answer:{row["answer"]}
      """
      documents.append(document)
   return documents