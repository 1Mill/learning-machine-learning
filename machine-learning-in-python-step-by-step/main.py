from matplotlib import pyplot
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print(dataset.shape) # * Describe shape
print('---')
print(dataset.head(20)) # * Peak at first 20 rows
print('---')
print(dataset.describe()) # * Calculate mean, min, max, etc. (basic statistics)
print('---')
print(dataset.groupby('class').size()) # * Count number of rows with each unique "class" name

# * box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show() # ! Does not work inside a Docker container
pyplot.savefig('./images/box-and-whiskers-plot.png')

# * histograms
dataset.hist()
pyplot.savefig('./images/histograms.png')

# * Multivariate plots
scatter_matrix(dataset)
pyplot.savefig('./images/scratter-plots.png')
