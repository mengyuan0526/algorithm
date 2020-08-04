#主成分判别法PCA
#使用decomposition库的PCA类选择特征
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris = load_iris()
iris.data
PCA(n_components=2).fit_transform(iris.data)

#线性判别法分析法（LDA）
#使用lda库的LDA类选择特征
from sklearn.lda import LDA
LDA(n_components=2).fit_transform(iris.data,iris.target)
