# deep-learning-challenge

This deep-learning challenge has not been as difficult as few others. I was able to do most of the work independently. I am glad that I am learning 
My instructors and class mates were able to help me with things I wasn't sure about. I was able to use ChatGPT as well in completing this challenge

Please see my Alphabet Soup analysis below.




Report on the Deep Learning Model for Alphabet Soup

Overview of the Analysis

The purpose of this analysis is to build and evaluate a deep learning model to predict the success of charity applications in the "Alphabet Soup" dataset. The dataset contains various features related to charity applications, such as the type of application, the classification of the organization, the amount of funds requested, and other factors. The target variable is whether the application was successful (IS_SUCCESSFUL), which is a binary classification task.

The deep learning model (a neural network) was trained to predict the target variable using multiple hidden layers and neurons. The model’s performance was evaluated based on its ability to predict success in charity applications accurately, and various optimization strategies were applied to improve model performance.

Results

Preprocessing of Data Before training the model, the following preprocessing steps were applied:
Categorical Encoding: Categorical variables such as APPLICATION_TYPE, CLASSIFICATION, and others were converted into numerical format using one-hot encoding. This is essential because neural networks require numerical input. Scaling: Numerical features like ASK_AMT, INCOME_AMT, etc., were scaled using StandardScaler to standardize the feature range, which helps the model converge faster. Handling Outliers: Outliers were detected using the IQR method and removed to prevent skewed model predictions.

Model Architecture The deep learning model was designed with the following architecture:
Input Layer: The input layer size corresponds to the number of features in the preprocessed data. Hidden Layers: The model had three hidden layers: First hidden layer: 128 neurons with ReLU activation. Second hidden layer: 64 neurons with LeakyReLU activation. Third hidden layer: 32 neurons with LeakyReLU activation. Output Layer: The output layer contained a single neuron with sigmoid activation for binary classification (predicting success or failure of applications).

Model Training The model was trained using the following parameters:
Epochs: 100 epochs to allow the model to learn from the data over multiple iterations. Batch Size: 32, which specifies the number of samples processed before the model’s weights are updated. Early Stopping: This callback was used to stop training if the validation loss did not improve after 10 epochs, preventing overfitting. Optimizer: Adam optimizer was used for efficient training.

Model Evaluation The model was evaluated using the test dataset (X_test_scaled, y_test), and the following results were obtained:
Test Accuracy: 75.2%

Test Loss: 0.485

These results show that the model performed reasonably well for predicting the success of charity applications, with a relatively high accuracy for a first attempt at a deep learning model.

Learning Curves The training and validation accuracy, as well as the loss curves, were plotted to visualize the model's learning process. Below are the key observations:
Training and Validation Accuracy:

The accuracy steadily increased over epochs. The gap between training and validation accuracy was small, suggesting that the model did not overfit the training data. Training and Validation Loss: Both the training and validation loss decreased over time, indicating that the model was learning and improving. The validation loss stabilized after some time, confirming that the model was not overfitting.

Hyperparameter Tuning and Optimization To improve model performance, several optimizations were applied:
More Neurons: The number of neurons in the hidden layers was increased to improve the model’s ability to capture more complex patterns. LeakyReLU: Instead of using ReLU in some layers, LeakyReLU was applied to avoid the vanishing gradient problem. Early Stopping: This was used to stop training once the validation loss stopped improving, preventing overfitting. Increased Epochs: The model was trained for a larger number of epochs (100) to allow the model to learn from the data fully. Summary of Key Results Test Accuracy: 75.2% — The model achieved over 75% accuracy, meeting the target for predictive accuracy. Test Loss: 0.485 — The relatively low test loss shows that the model made reasonable predictions.

Key observations from the learning curves:

Both training and validation accuracy increased steadily. Training and validation loss decreased, indicating the model learned to minimize errors. Visual Results: Training vs. Validation Accuracy:

Training vs. Validation Loss:

Challenges and Improvements: Overfitting: Although the model performed well, there is still room to improve by tuning the architecture and adjusting the learning rate or using more advanced techniques like dropout. Hyperparameter Tuning: Further optimization of hyperparameters, including the learning rate, batch size, and the number of hidden layers/neurons, could further improve performance.

Conclusion

The deep learning model for predicting charity application success was successful in achieving a test accuracy of 75.2%, meeting the target goal. Several optimizations, including increased network complexity, early stopping, and scaling, contributed to the model's performance. Further refinements, such as additional hyperparameter tuning and improved feature engineering, could boost the model's predictive accuracy even further.


