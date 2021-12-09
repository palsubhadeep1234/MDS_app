#Importing necessary packages
import streamlit as st
import numpy as np
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt

st.title('Predicting the Assigned Position of NBA Players')
st.image("./introduction_nba.png")

st.write('__Five important player positions__')
st.image("./player_position.png")

#Defining font sizez
st.markdown("""
<style>
.bfont {
    font-size:40px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.mfont {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

#Writing maximum and minimum values to normalize player input stats

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

#Getting inputs from users

st.write('<p class = "bfont">Enter player statistics</p>', unsafe_allow_html=True)

st.write('<p class = "mfont">Height (in inch)</p>', unsafe_allow_html=True)
height = st.slider('', max_height, min_height, step=0.1)

st.write('<p class = "mfont">Number of assists</p>', unsafe_allow_html=True)
assist = st.slider('', max_assist, min_assist, step=1)

st.write('<p class = "mfont">Number of blocks</p>', unsafe_allow_html=True)
block = st.slider('', max_block, min_block, step=1)

st.write('<p class = "mfont">Number of 2-pointers scored</p>', unsafe_allow_html=True)
two_pnt = st.slider('', max_two_pnt, min_two_pnt, step=1)

st.write('<p class = "mfont">Number of 3-pointers scored</p>', unsafe_allow_html=True)
three_pnt = st.slider('', max_three_pnt, min_three_pnt, step=1)

st.write('<p class = "mfont">Number of personal fouls</p>', unsafe_allow_html=True)
foul = st.slider('', max_foul, min_foul, step=1)

st.write('<p class = "mfont">Number of total rebounds</p>', unsafe_allow_html=True)
rebound = st.slider('', max_rebound, min_rebound, step=1)


#Loading the NN model and weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("MDS_model.h5")

#Normalizing input data
height_norm = (height-min_height)/(max_height - min_height);
assist_norm = (assist/(max_assist - min_assist));
block_norm = (block/(max_block - min_block));
two_pnt_norm = (two_pnt/(max_two_pnt - min_two_pnt));
three_pnt_norm = (three_pnt/(max_three_pnt - min_three_pnt));
foul_norm = (foul/(max_foul - min_foul));
rebound_norm = (rebound/(max_rebound - min_rebound));

test_data = np.zeros((1,7), dtype = float);
test_data[0,:] = [height_norm, assist_norm, block_norm, two_pnt_norm, three_pnt_norm, foul_norm, rebound_norm];

#Predicting output value using trained weights
y_predict = loaded_model.predict(test_data);
#st.dataframe(test_data)
#st.dataframe(y_predict)

#Plotting probability of player position
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

#st.dataframe(y_pred_process)

pos = ['Center', 'Power Forward', "Small Forward", "Point Guard", "Shooting Guard"];
no_prediction = np.count_nonzero(y_pred_process == 1)

#Printing most probable player position

index = [];
if  np.count_nonzero(test_data == 0) > 5:
    st.markdown(f"""PLease enter player statistics""")
elif no_prediction == 1:
    for i in range(0,5):
        if y_pred_process[0,i] == 1:
            index.append(i);
    # st.markdown(f"""The player can play in __*{pos[index[0]]}*__ position""")
    st.markdown('<p class = "mfont"> The player can play in</p>' var(pos[index[0]]), unsafe_allow_html=True)
elif no_prediction == 0:
    st.markdown(f"""There is no distinct player position for given statistics""")
else:
    for i in range(0,5):
        if y_pred_process[0,i] == 1:
            index.append(i);
    st.markdown(f"""The player can play either in __*{pos[index[1]]}*__ or __*{pos[index[0]]}*__ position""")