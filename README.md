MNIST dataset consists of pair, “handwritten digit image” and “label”. Digit ranges from 0 to 9, meaning 10 patterns in total.

handwritten digit image: This is gray scale image with size 28 x 28 pixel.

label : This is actual digit number this handwritten digit image represents. It is either  0 to 9.

![MNIST-photo](http://corochann.com/wp-content/uploads/2017/02/mnist_plot.png "Taken from :http://corochann.com/wp-content/uploads/2017/02/mnist_plot.png" )

## PCA files

MNIST dataset is loaded and split into a training set and a test set (the first 60 000 instances for training and the remaining 10 000 for testing)

Random Forest classifier is trained on the dataset and time is measured to see how long it takes, then the model is evaluated on the test set.

PCA is used to reduce the dataset’s dimensionality, with an explained variance ratio of 95%. New Random Forest classifier is trained on the reduced dataset and time is measured.

 Main goal here is to observe the difference between the original dataset and reduced dataset's performances.
 
 ## Ensemble files
 
MNIST dataset is loaded and split into a training set, validation set and a test set (50 000 instances for training, 10 000 for validation, 10 000 for testing) 

The dataset is actually ordered for 60 000 instances for training when we take the last 10 000 for validation it doesn't include a set from all classes. Therefore, we need to take a 1 000 instances from in each 6 0000. By this way, we will have sets include data from all classes. 

Initially, Multinomial Naïve Bayes, Random Forest and SVM classifiers are used to train the model with training dataset. Then they performed on validation set. 

Hard voting classifier is used to ensemble these classifiers. It is performed on validation set too in order to compare the accuracy results. 

Because of SVM is resulted the best, it is performed on the test set.

## K - means files

MNIST dataset is loaded and all of them is used as a trainin set. 

K-means clustering is perforemd on the training data with k=10 by assuming that the data is unsupervised. The accuracy is computed by a for loop. It checks the prediction and the actual label. 

Then, PCA is used to reduce the dataset’s dimensionality, with an explained variance ratio of 85%. K-means clustering is perforemd again on reduced dataset with k=10. Accuracy is computed by the same way.


If you have any questions, problems or any advice, feel free to contact : irem.karaoglu.tedu@gmail.com
