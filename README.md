# Music Instrument classification
The classification of musical instruments using supervised learning is studied. A combined feature set including psychoacoustically relevant spectral features is used.

## Understanding the Dataset
The dataset contains audio files of .mp3 format 20 instruments, total of 13,682 audio sounds.

Data Distribution:
![data distribution](https://user-images.githubusercontent.com/66205648/158135220-ff4e7417-ba72-4489-9629-8680fbc5bd64.png)

New dataset is the csv file with extracted features for all 20 instruments with total columns of 28 including labels.

## Preprocessing and Feature Extraction
Features are extracted using the library librosa.

- **chroma_stft** - representation of entire range of frequencies in terms of 12 different pitch classes

- **rms** - average ampltitude over a time period

- **spectral centroid** - average of all frequencies of sound

- **spectral bandwidth** - gap in the lowest and the highest frequncy

- **spectral rolloff** - the qth frequency below which certain range of frequencies is found

- **zero crossing rate** - number of times amplitude of audio passes through zero

- **mfcc** - coefficient that represent spectrum across time


## EDA
**Introduction:**

- **merged_data** dataset comprises of 13682 rows and 28 columns.
- Dataset comprises of continious variable and float data type. 
- Dataset column varaibales 'Label', 'chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean',
       'spectral_bandwidth_mean',
       'spectral_rolloff_mean', 'zero_crossing_rate_mean', and mfcc mean coefficients

**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features
![desc](https://user-images.githubusercontent.com/66205648/158571983-a2bc34d4-ad95-46b4-90b4-27da344827aa.png)

**Visualisation of Variables:**

- Scatter plot
- Distplot
- Kernel density plot

**Analysis we got**
- rms_mean is highly correlated with mfcc_mean_0
- spectral_rolloff_mean, spectral_bandwidth_mean, zero_crossing_rate_mean is highly correlated with spectral_centroid_mean
- Double-bass has the highest chroma value
- Contrabassoon has the highest rms value

Before exploratory data analysisand after splitting we scaled the data using standardization to shift the distribution to have a mean of zero and a standard deviation of one.
```
from sklearn.preprocessing import StandardScaler

scaler1 = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features1 = pd.DataFrame(scaler1.fit_transform(df[['spectral_centroid_mean', 'zero_crossing_rate_mean']]))
```

**PCA()** - Principal Component Analysis is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation. These new transformed features are called the Principal Components.
```
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
data_pca = pca.fit_transform(filtered_mfcc)
pca.explained_variance_ratio_
data_pca = pd.DataFrame(data_pca)
data_pca["Label"] = data["Label"]
data_pca.columns = ['PCA1', 'PCA2', "Label"]
data_pca
```

**TSNE()** - T-distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions.
 
```
from sklearn.manifold import TSNE
tsne = TSNE()
data_tsne = tsne.fit_transform(filtered_mfcc)
```

**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.


## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.
We got the highest accuracy of 96 percent using Logistic Regression Model.
#### Support Vector machine
- Support Vector Machine(SVM) is a supervised machine learning algorithm used for both classification and regression. 
- The objective of SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. 
- The dimension of the hyperplane depends upon the number of features. If the number of input features is two, then the hyperplane is just a line. If the number of input features is three, then the hyperplane becomes a 2-D plane. 
We got the highest accuracy of 97 percent using SVM() model.
## Deployment

### Flask 
We also create our app.
- The app runs on local host.# Music Instrument classification
The classification of musical instruments using supervised learning is studied. A combined feature set including psychoacoustically relevant spectral features is used.
