import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
# Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.image('https://i.pinimg.com/originals/9a/c6/76/9ac676d82df3faa7756069885fec9e36.gif',width = 700)
st.header('Model of housing prices to predict median house values in California',divider = True)

# st.subheader('''User Nust Enter Given values to predict
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://blog.architizer.com/wp-content/uploads/Untitled-design.gif')

temp_df = pd.read_csv('california.csv')

random.seed(15)
all_value = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} range',int(min_value),int(max_value),random.randint(int(min_value),int(max_value)))

    all_value.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_value])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

import time

st.write(pd.DataFrame(dict(zip(col,all_value)),index = [1]))
progrees_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price!!')
place = st.empty()
place.image('https://content.presentermedia.com/files/animsp/00002000/2470/stick_figure_search_clues_anim_lg_wm.gif',width = 80)
if price>0:
    for i in range(100):
        time.sleep(0.01)
        progrees_bar.progress(i+1)
    body = f'Predicted Median House Price: ${price.round(2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Invalid House features value'
    st.warning(body)
    
    