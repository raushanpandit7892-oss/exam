import numpy as np
from keras.models import Sequential
from keras.layers import Dense
X = np.random.rand(10, 3)  # Random data with 3 input features
y = np.random.randint(0, 2, (10, 1))  # Random binary labels (0 or 1)
model = Sequential()
model.add(Dense(units=4, activation='relu', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=2)
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')