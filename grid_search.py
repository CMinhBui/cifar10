import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from preact_resnet_model import ResNet_model
import utils

def create_model(res_layer_params=(3, 32, 25), reg=0.0001, optimizer='adam'):
    model = ResNet_model(res_layer_params=res_layer_params, reg=reg)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

(X_train, y_train), (X_test, y_test) = utils.load_cifar(normalize=True)

model = KerasClassifier(build_fn=create_model, verbose=2)
batch_size = [32, 64, 128]
epochs = [50, 100, 200]
res_layer_params = [(3, 32, 25), (3, 32, 10)]
reg = [0.0001, 0.0005, 0.001, 0.005]
optimizer = ['SGD', 'RMSprop', 'Adam']

params_grid=dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, res_layer_params=res_layer_params, reg=reg)
grid = GridSearchCV(estimator=model, param_grid=params_grid, cv=5, n_jobs=8)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))