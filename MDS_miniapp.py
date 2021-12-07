from imghdr import tests
import streamlit as st
import numpy as np
pip install tensorflow
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt

st.title('Predicting the Assigned Position of NBA Players')
st.image("./nba_poster.jpg", width=500, clamp="bottom")

max_height = 87.0;
min_height = 69.0;

max_playtime = 60;
min_playtime = 0;

max_assist = 25;
min_assist = 0;

max_block = 12;
min_block = 0;

max_two_pnt = 34;
min_two_pnt = 0;

max_three_pnt = 13;
min_three_pnt = 0;

max_foul = 6;
min_foul = 0;

max_rebound = 30;
min_rebound = 0;

st.write('__Input player statistics__')
height = st.slider('Height (in inch)', max_height, min_height, step=0.1)
playtime = st.slider('Playtime (in min)', max_playtime, min_playtime, step=1)
assist = st.slider('Number of assist', max_assist, min_assist, step=1)
block = st.slider('Number of block', max_block, min_block, step=1)
two_pnt = st.slider('Number of 2-pointers scored', max_two_pnt, min_two_pnt, step=1)
three_pnt = st.slider('Number of 3-pointers scored', max_three_pnt, min_three_pnt, step=1)
foul = st.slider('Number of personal foul', max_foul, min_foul, step=1)
rebound = st.slider('Number of total rebound', max_rebound, min_rebound, step=1)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("MDS_model.h5")

height_norm = (height-min_height)/(max_height - min_height);
assist_norm = (assist/(max_assist - min_assist));
block_norm = (block/(max_block - min_block));
two_pnt_norm = (two_pnt/(max_two_pnt - min_two_pnt));
three_pnt_norm = (three_pnt/(max_three_pnt - min_three_pnt));
foul_norm = (foul/(max_foul - min_foul));
rebound_norm = (rebound/(max_rebound - min_rebound));

test_data = np.zeros((1,7), dtype = float);
test_data[0,:] = [height_norm, assist_norm, block_norm, two_pnt_norm, three_pnt_norm, foul_norm, rebound_norm];

y_predict = loaded_model.predict(test_data);
st.dataframe(test_data)
st.dataframe(y_predict)

plot_data = {'C':y_predict[0,0], 
             'PF':y_predict[0,1],
             'SF':y_predict[0,2],
             'PG':y_predict[0,3],
             'SG':y_predict[0,4]}
pos = list(plot_data.keys())
count = list(plot_data.values())
  
fig = plt.figure(figsize = (10, 8))
 
# creating the bar plot
plt.bar(pos, count, color ='black', width = 0.5)
 
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("Position prediction")
plt.savefig('position_prediction.png')
#plt.show()

st.image("./position_prediction.png");

y_pred_process = np.ones((1,5), dtype=int);
for i in range(0,5):
    if (y_predict[0,i] < 0.4):
        y_pred_process[0,i] = 0;

st.dataframe(y_pred_process)

pos = ['Center', 'Power Forward', "Small Forward", "Point Guard", "Shooting Guard"];
no_prediction = np.count_nonzero(y_pred_process == 1)

index = [];
if no_prediction == 1:
    for i in range(0,5):
        if y_pred_process[0,i] == 1:
            index.append(i);
    st.markdown(f"""The player can play in __*{pos[index[0]]}*__ position""")
else:
    for i in range(0,5):
        if y_pred_process[0,i] == 1:
            index.append(i);
    st.markdown(f"""The player can play either in __*{pos[index[1]]}*__ or __*{pos[index[0]]}*__ position""")



st.markdown(f"""Number of prediction = {no_prediction}""")
