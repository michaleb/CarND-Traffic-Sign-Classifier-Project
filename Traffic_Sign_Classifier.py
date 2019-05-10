import cv2
import glob
import pickle 
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
get_ipython().magic('matplotlib inline')


training_file = '../data/train.p'
validation_file = '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))


# TODO: Number of training examples
n_train = len(np.array(X_train))

# TODO: Number of validation examples
n_valid = len(np.array(X_valid))

# TODO: Number of testing examples.
n_test = len(np.array(X_test))

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))


print("Number of training examples =", n_train)
print('Number of validating examples =', n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of unique classes =", n_classes)


# Include an exploratory visualization of the dataset
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(4,4))
#plt.imshow(image)
print('Example image from Training set')
plt.title('Image shape '+ str(image.shape), fontsize=15)
plt.xlabel('Traffic Sign Class: '+str(y_train[index]), fontsize=15)
plt.imshow(image)


#create function to display number of signs per class
def show_images(X_data, y_data):
    
    unique_class = np.unique(y_data)
    plt.figure(figsize=(25,25))

    for class_id in unique_class:
        #groups all the indices in the set of images where the image label matches each unique class label
        index = np.where(y_data == class_id)[0] 

        #selects the first image from the newly create index of images for each class
        image = X_data[index[0]]

        plt.subplot(7, 7, class_id+1)
        plt.axis('off')
        plt.title("class {0} total({1})".format(class_id, len(index)), fontsize=18)

        plt.imshow(image)

    plt.show()
    return


#Show distribution of classes over the training set
plt.hist(y_train, 43, color='green', width=2, align='mid')
plt.ylabel('No. of Traffic Signs / Class', fontsize=15)
plt.xlabel('Traffic Sign Classes', fontsize=15)
plt.title('Training Set Distribution', fontsize=20)
plt.show()

show_images(X_train, y_train)

#Show distribution of classes over the validation set
plt.hist(y_valid, 43, color='red', width=2, align='mid')
plt.ylabel('No. of Traffic Signs / Class', fontsize=15)
plt.xlabel('Traffic Sign Classes', fontsize=15)
plt.title('Validation Set Distribution', fontsize=20)
plt.show()

show_images(X_valid, y_valid)

#Show distribution of classes over the test set
plt.hist(y_test, 43, color='blue', width=2, align='mid')
plt.ylabel('No. of Traffic Signs / Class', fontsize=15)
plt.xlabel('Traffic Sign Classes', fontsize=15)
plt.title('Test Set Distribution', fontsize=20)
plt.show()

show_images(X_test, y_test)



# Preprocess the data here. It is required to normalize the data. 

def preprocess(image_data):
    """
    Converts the rgb image data to grayscale and normalizes to a range of [-1, 1]
    and reshapes image data to[..32x32x1]
    :param image_data: The image data to be converted and normalized
    :return: Normalized image data
    """
    gray = np.dot(image_data[:,:,:,:3], [0.299, 0.587, 0.114])
    norm = (gray.astype(float) - 128)/128
    norm = norm.reshape(norm.shape + (1,)) 
    
    return norm


X_train = preprocess(X_train) 
X_valid = preprocess(X_valid)
X_test  = preprocess(X_test)


#display a sample of images in training set before preprocessing
print('Shape of traffic signs',train['features'][0].shape)
plt.figure(figsize=(25,25))
index = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500]

for i in range(len(index)):
    plt.subplot(1,10, i+1)
    plt.imshow(train['features'][index[i]].squeeze())
plt.show()    

#display a sample of Processed images in training set
print('Shape of processed traffic signs',X_train[0].shape)
plt.figure(figsize=(25,25))
index = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500]

for i in range(len(index)):
    plt.subplot(1,10, i+1)
    plt.imshow(X_train[index[i]].squeeze(), cmap='gray')
    
plt.show()   

X_train, y_train = shuffle(X_train, y_train) 


# LeNet Model Architecture

EPOCHS = 20
BATCH_SIZE = 128

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


# Train, Validate and Test the Model

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75}) 
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './Traffic_Sign_Classifier')

#restores all parameters, weights, biases etc.., created and used in training step
#to evaluate performance of the model on a set of 'unknown' test images

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate(X_train, y_train)
    print("Train Accuracy = {:.3f}".format(train_accuracy))

#restores all parameters, weights, biases etc.., created and used in training step
#to evaluate performance of the model on a set of 'unknown' test images

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# Load the images and plot them here.

#loads and displays new images. Images are of various sizes and require processing
new_images = glob.glob('../data/new_test_images/ts*.jpeg')
plt.figure(figsize=(25,25))

new_test_img = []
images = []
for i, fname in enumerate(new_images):
    image = plt.imread(fname)
    images.append(image)
    image_res = cv2.resize(image, (32,32))
    new_test_img.append(image_res)
    plt.subplot(1,5, i+1)
    plt.imshow(image)

plt.show()    

# Predict the Sign Type for each new image
new_test_img = np.array(new_test_img)
plt.figure(figsize=(20,20))
new_X_test = []

for i, image in enumerate(new_test_img):
    gray = np.dot(image[:,:,:3], [0.299, 0.587, 0.114])  #convert image to grayscale
    norm = (gray.astype(float) - 128)/128                #normalize grayscale to range [-1, 1]
    
    img_res = norm.reshape(norm.shape + (1,))            #reshape image from (32x32) to (32x32x1)
    new_X_test.append(img_res)
    plt.subplot(1,5, i+1)
    plt.imshow(norm, cmap='gray')
    
print("Shape of processed images",format(new_X_test[0].shape))
plt.show()

#Predicting the class of the 5 new images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    plt.figure(figsize = (25,25))
    predict = sess.run(logits, feed_dict={x: new_X_test, keep_prob: 1})
    prediction = tf.argmax(predict, 1)
        
    for i in range(len(images)):
        plt.subplot(1,5, i+1)
        plt.title("Predicted Class: {0} ".format(sess.run(prediction[i])), fontsize=20)
        
        plt.imshow(images[i])
    
    plt.show()    

# Calculate the accuracy for these 5 new images. 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    #Ground truth/actual class to which the new test images belong.
    new_y_test = [33, 11, 13, 18, 16]         
    prediction = tf.argmax(predict, 1)
    
    correct = 0
    for i in range(len(images)):
        if new_y_test[i] == sess.run(prediction[i]):
            correct += 1
    accuracy = correct/len(images)
    print('Accuracy of model =', accuracy)

# Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    print('\n','Top 5 Softmax probabilities','\n')
    output = sess.run(tf.nn.softmax(predict))
    print(sess.run(tf.nn.top_k(tf.constant(output), k=5)))

# Visualize your network's feature maps.

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

