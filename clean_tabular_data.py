from unicodedata import category
import pandas as pd

def Products_cleaner():
    #init
    df = pd.read_csv("Products.csv", lineterminator="\n")
    #convert price to float
    df['price'] = df['price'].str.replace(',', '')
    df['price'] = df['price'].str.strip('£').astype(float)
    #drop unneeded column
    df=df.drop('Unnamed: 0', axis=1)
    #clean extra details from product_name column
    df['product_name']= df['product_name'].str.split('|').str[0]
    #remove emojis from product_name and description column
    df['product_name']= df['product_name'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    df['product_description']= df['product_description'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    #create category_id column with cat.codes
    df['category'] = df['category'].str.split(' / ').str[0]
    df['category'] = df['category'].str.strip()
    df['category_id'] = df['category'].str.split(' / ').str[0].astype('category').cat.codes.astype('int64')
    #save dataframe to a new file
    #print(df.head())
    #df.to_csv('Cleaned_Products.csv')

def DF_merger():
    df = pd.read_csv("Cleaned_Products.csv", lineterminator="\n")
    df_img = pd.read_csv("Images.csv", lineterminator="\n")
    df_img = df_img.rename(columns={'id':'image_id'})
    df_merge = pd.merge(df, df_img[['product_id', 'image_id']], left_on='id', right_on='product_id')
    df_merge = df_merge.drop(columns=['Unnamed: 0', 'product_id'])
    df_merge = df_merge[['id', 'image_id', 'product_name', 'category', 'product_description', 'price', 'location', 'category_id\r']]
    #print(df_merge['category'].tail())
    df_merge.to_csv('Image+Products.csv')

#Products_cleaner()
#DF_merger()