# MDS_app
The mechanistic data science methodology was used to generate a neural network for predicting an NBA player’s assigned 
position based on their box score. Using linear regression, a correlation heat map, and histograms broken down by position, 
we determined that among the 24 statistics considered, blocks, assists, turnovers, personal fouls, 2 pointers made, 3 pointers 
made, and height showed the most distinct variation between positions. Data was collected for each individual player’s 
box score over six seasons and used to train a neural network with two hidden nodes. Leaky Relu and softmax activation 
functions were used in order to assign a probability to each player position based on the 7 input parameters. The 
network achieved a 62% perfect match accuracy in predicting position on the first guess, and a 77% half match accuracy.
Using the weights from the trained NN model, we created a web-based application, named ”MDS_miniapp”, hosted in 
streamlit.io where users can enter seven important player statistics to view an output probability for each of the five positions.

Results: 
62% accuracy in choosing out of the 5 NBA positions correctly.
77% accuracy with 2 guesses.

Challenges:
Primary box scores statistics may not be enough. 
A single player's box score over 20 minutes is not enough.
