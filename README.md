# Image Recognition to Identify Species of Flowers

## 1.0 Abstract
This report documents the development of a Convolutional Neural Network (CNN) Model using TensorFlow's "tf_flowers" dataset to classify species of flowers. The dataset consists of 3,670 images categorized into five species: Daisy, Dandelion, Roses, Sunflowers, and Tulips. This study details the CNN's model architecture, hyperparameter tuning, and regularization techniques, achieving a final model accuracy of 83%. The report highlights the importance of careful hyperparameter tuning and suggests directions for future research.

## 2.0 Introduction
Image classification is a significant challenge in artificial intelligence, particularly with large datasets where traditional methods become impractical. CNNs address this by automating feature extraction, handling large datasets, and recognizing complex patterns. This report focuses on the capabilities of CNNs to classify different flower species within the "tf_flowers" dataset.

## 3.0 Literature Review
Hiary et al. (2018) demonstrated the effectiveness of CNNs in distinguishing flower species, highlighting the model's ability to handle variations in color, shape, and contextual elements like leaves and grass. Their findings support the high accuracy and superior performance of CNNs over traditional classification methods, underscoring the potential of CNNs in complex image classification tasks.

## 4.0 Methodology

### 4.1 Data Description
The dataset includes 3,670 images split into training (70%), validation (15%), and testing (15%) sets, across five different flower species.

### 4.2 CNN Model

#### 4.2.1 Architecture
The model uses TensorFlow's Keras API with a sequential layout, incorporating:
- Multiple layers with 32 filters of size 3x3 for initial feature extraction.
- ReLU activation function to prevent vanishing gradients.
- SoftMax activation for output.
- MaxPooling of 2x2 to reduce dimensionality.

#### 4.2.2 Regularization Techniques
Dropout rates between 0.5 and 0.2 and optional batch normalization were implemented to combat overfitting and stabilize training.

#### 4.2.3 Training and Evaluation
The model was trained using the Adam optimizer over 20 epochs with a batch size of 32. Performance metrics included accuracy, loss, confusion matrix, and classification reports.

## 5.0 Results

### 5.1 Baseline Model
The baseline model, with a dropout of 0.5, achieved an initial accuracy of 80%. However, signs of overfitting were observed, indicating the need for parameter adjustments.

### 5.2 Impact of Batch Size
Increasing the batch size from 64 to 128 reduced accuracy to 71%, suggesting that smaller batch sizes are more effective for this dataset.

### 5.3 Adjusting Dropout and Learning Rate
Adjustments in the learning rate and dropout rate led to varied results:
- A lower learning rate of 0.0001 with a dropout of 0.2 increased accuracy to 83%.
- Further tests adjusting these parameters showed a drop in performance when batch normalization was applied, reducing accuracy to 80% and introducing some overfitting.

### 5.4 Pooling Method
Changing from MaxPooling to AveragePooling decreased accuracy to 74%, indicating that MaxPooling was more effective.

## 6.0 Discussion
The optimal model configuration achieved an accuracy of 83% with a learning rate of 0.001 and a dropout rate of 0.2, using a batch size of 128. This setup effectively balanced learning and regularization, enhancing overall model performance.

## 7.0 Conclusion
Through various experiments, the CNN model's accuracy improved from an 80% baseline to 83%, with reduced overfitting. These results contribute to deeper insights into CNN applications in image recognition and provide a foundation for further exploration of more sophisticated models.

## 8.0 References
1. Hiary, H., Saadeh, H., Saadeh, M., & Yaqub, M. (2018). Flower classification using deep convolutional neural networks. IET Computer Vision, 12(6), 855-862.
2. Singh, A., & Singh, P. (2020). Image Classification: A Survey. Journal of Informatics Electrical and Electronics Engineering, 1(2), 1-9.
