from function import *

# main
train.set = imageset(data=train)
test.set = imageset(data=test)
eig = Eigen()

avg = average_image(data=train)

cov = covariance_matrix(data=train, average=avg, eigen=eig)

eig.values, eig.vectors, eig.index = get_eigen(covariance=cov)
eig.values = eig.values[: eig.index]
eig.vectors = eig.vectors[:, : eig.index]

trans = translation_matrix(eigen=eig)

test.feats = feature_values(data=test, average=avg, translation=trans)
train.feats = feature_values(data=train, average=avg, translation=trans)

command_input()

