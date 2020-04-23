import os
import tensorflow as tf
import pandas as pd
import calendar;
import time;


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

books = pd.read_csv(r".\Dataset\BX-CSV-Dump\BX-Books.csv", sep=";", encoding='latin1')
ratings = pd.read_csv(r".\Dataset\BX-CSV-Dump\BX-Book-Ratings.csv", sep=";",
                      encoding='latin1')
users = pd.read_csv(r".\Dataset\BX-CSV-Dump\BX-Users.csv", sep=";",
                    encoding='latin1')
full_data_df = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
useful_data_df = full_data_df[['Location', 'Age', 'Book-Author', 'Year-Of-Publication', 'Book-Rating']]
loc_df = pd.DataFrame([x.split(', ', 2)[0:2] for x in useful_data_df['Location'].tolist()])
loc_df.columns = ['City', 'State']
useful_data_df = pd.concat([useful_data_df, loc_df], axis=1).drop('Location', axis=1)
'''
le = LabelEncoder()
useful_data_df['Book-Author-Enc'] = le.fit_transform(useful_data_df['Book-Author'].values).reshape(-1, 1)
'''
le = LabelEncoder()
useful_data_df['City-Enc'] = le.fit_transform(useful_data_df['City'].values).reshape(-1, 1)

le = LabelEncoder()
useful_data_df['State-Enc'] = le.fit_transform(useful_data_df['State'].values).reshape(-1, 1)

useful_data_df = useful_data_df.drop(['Book-Author', 'City', 'State'], axis=1)
X, y = useful_data_df.drop('Book-Rating', axis=1), useful_data_df[['Book-Rating']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

RUN_NAME = calendar.timegm(time.gmtime())

learning_rate = 0.01
training_epochs = 100

number_of_inputs = 4

number_of_outputs = 1

layer_1_nodes = 20
layer_2_nodes = 100
layer_3_nodes = 50

input_x = tf.placeholder(tf.float32, shape=(None, number_of_inputs), name='Input_X')
weights_1 = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
biases_1 = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
layer_1_output = tf.nn.relu(tf.matmul(input_x, weights_1) + biases_1)

weights_2 = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
biases_2 = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights_2) + biases_2)

weights_3 = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
biases_3 = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights_3) + biases_3)

weights_o = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs],
                            initializer=tf.contrib.layers.xavier_initializer())
biases_o = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
prediction = tf.matmul(layer_3_output, weights_o) + biases_o

actual_y = tf.placeholder(tf.float32, shape=(None, 1), name="Actual_Y")
cost = tf.reduce_mean(tf.squared_difference(prediction, actual_y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

with tf.Session() as session:
    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    training_writer = tf.summary.FileWriter("./logs/{}/training".format(RUN_NAME), session.graph)
    testing_writer = tf.summary.FileWriter("./logs/{}/testing".format(RUN_NAME), session.graph)

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())
    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):
        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={input_x: X_train, actual_y: y_train})
        if epoch % 5 == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            training_cost, training_summary = session.run([cost, summary],
                                                          feed_dict={input_x: X_train, actual_y: y_train})
            testing_cost, testing_summary = session.run([cost, summary],
                                                        feed_dict={input_x: X_train, actual_y: y_train})
            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))
    testing_cost = session.run(optimizer, feed_dict={input_x: X_test, actual_y: y_test})
    print("Testing Cost: {}".format(testing_cost))
