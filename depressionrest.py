import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from flask import Flask,Response,request
import logging
import json
logging.info("Logging is happening")
le= None
app= Flask(__name__)




@app.route('/postbus',methods=['POST'])
def retpostResponse():
    df=pd.read_json(json.dumps(request.json['REQUEST']),orient='records')
    data=df[['POSTS']]
    col="POSTS"
    print("sam")
    print(data[col].values[0])

    df['words_per_comment'] = df['POSTS'].apply(lambda x: len(x.split()))
    print(df['words_per_comment'])
    print(df['POSTS'].apply(lambda x: len(x.split())))
    df['http_per_comment'] = df['POSTS'].apply(lambda x: x.count('http'))
    df['music_per_comment'] = df['POSTS'].apply(lambda x: x.count('music'))
    df['question_per_comment'] = df['POSTS'].apply(lambda x: x.count('?'))
    df['img_per_comment'] = df['POSTS'].apply(lambda x: x.count('jpg'))
    df['excl_per_comment'] = df['POSTS'].apply(lambda x: x.count('!'))
    df['ellipsis_per_comment'] = df['POSTS'].apply(lambda x: x.count('...'))
    X_test,y_test=factorize_sel(df)
    print(df)
    #1
    # with open('enc_t1.pkl','rb') as handle:
    #     model=pickle.load(handle)
    #     model_dict=dict(zip(model.classes_,model.transform(model.classes_)))
    # df["GVNG_CLS_CD"]=df["GVNG_CLS_CD"].apply(lambda x: model_dict.get(x,-11111))
    # print( df["GVNG_CLS_CD"])
    # #2
    # with open('enc_t2.pkl','rb') as handle:
    #     model=pickle.load(handle)
    #     model_dict=dict(zip(model.classes_,model.transform(model.classes_)))
    # df["COV_LST"]=df["COV_LST"].apply(lambda x: model_dict.get(x,-11111))
    # print( df["COV_LST"])
    # #3
    # with open('enc_t3.pkl','rb') as handle:
    #     model=pickle.load(handle)
    #     model_dict=dict(zip(model.classes_,model.transform(model.classes_)))
    # df["PDM_ST_CD"]=df["PDM_ST_CD"].apply(lambda x: model_dict.get(x,-11111))
    # print( df["PDM_ST_CD"])
    
    # df=df[['GVNG_CLS_CD','GL_CLSS_CD','COV_LST','PDM_ST_CD','TOT_TERM_EXP_AMT','SEL_CODE']]
    # print(df)

    # X_test,y_test=factorize_sel(df)

    with open('model/RandomForest.pkl','rb') as handle:
        model=pickle.load(handle)
    print(X_test)
    y_pred=model.predict(X_test)
    print(y_pred)
    response_text=''
    response_text=str(y_pred)
    print('{"MODEL_RESPONSE:"'+response_text+'"}')
    respModel= Response('{"MODEL_RESPONSE:"'+response_text+'"}',status=200,mimetype='application/json')
    respModel.headers['Content-Type']='application/json'


    return respModel
@app.route('/getbus',methods=['GET'])
def retgetResponse():
    username=request.args.get('username')
    print ("username")
    return username

def factorize_sel(data):
    y=data['POSTS']
    data=data.drop('POSTS',1)
    return data.as_matrix().astype(int),y.as_matrix()
if __name__ == "__main__":
    app.run(host='',port=3000)