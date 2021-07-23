
"""
逻辑回归算法
基本含义：https://easyai.tech/ai-definition/logistic-regression/#what
具体算法：https://zhuanlan.zhihu.com/p/54290246
"""

from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

# 1.加载数据
iris = datasets.load_iris()
X = iris.data[:, :2]  # 使用前两个特征
Y = iris.target

# 2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 设置随机数种子，以便比较结果。

# 3.标准化特征值
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
    penalty：惩罚项的种类，l1或l2。一般选择l2正则，如果效果不好；可以考虑l1正则；或者使用l1正则来进行特征筛选过程。
    C：惩罚系数，控制惩罚程度。值越小，正则化程度越强，是正则化系数的倒数。
    class_weight：考虑类不平衡，代价敏感。当class_weight为balanced时会自动进行权重计算。
    random_state：随机种子的设置，数据混洗实用的随机种子。因为sag和liblinear都是随机平均年梯度下降
                  都是使用一部分样本来计算梯度，所以不同随机种子对这两个策略有影响。
    max_iter：算法收敛的最大迭代次数，默认是100。
    tol：收敛条件，默认是0.0001，也就是只需要收敛的时候两步之差小于0.0001就停止。
    verbose：是否会输出一些模型运算过程中的东西。
    warm_start：是否用上次模型结果进行初始化。
    dual：用来指明是否将原问题改成他的对偶问题，即相反问题。
'''

# 4. 训练逻辑回归模型
logreg = linear_model.LogisticRegression(penalty='l1', C=1e5, solver='liblinear')
logreg.fit(X_train, Y_train)

# 5. 预测
prepro = logreg.predict_proba(X_test_std)
acc = logreg.score(X_test_std, Y_test)

print(f"acc: {acc}")
