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

# * Part 2 - Evaluating some algorithms
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print('---')
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s, %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.clf() # ! Clear plot of previous state
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.savefig('./images/algorithm-comparison.png')

# * Part 3 - Make Predictions

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print('---')
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
