#Kutuphaneleri yukle
import pandas as pd
import keras

# Veri seti yukle 
dataset= pd.read_csv('KronikBobrekHastaligi.csv')

dataset_x=dataset.iloc[:,1:21].values
dataset_y=dataset.iloc[:,21].values

#Veri setini eğitim ve test olarak bölme
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(dataset_x,dataset_y,test_size=0.30, random_state=17)

#Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#YSA'yı oluşturma
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

#Girdi katmanı ve ilk gizli katmanımızı oluşturma
classifier.add(Dense(kernel_initializer = 'normal', input_dim = 20, units = 9, activation = 'relu'))

#İkinci gizli katman
classifier.add(Dense(kernel_initializer = 'normal', units = 8, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'normal', units = 7, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'normal', units = 6, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'normal', units = 5, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'normal', units = 5, activation = 'relu'))

#Çıktı katmanı
from keras.layers import Dense,Dropout
classifier.add(Dropout(0.3)) 
classifier.add(Dense(kernel_initializer = 'uniform', units = 1,  activation = 'sigmoid'))

#YSA derleme
#binary_crossentropy; çıktımız binary olduğu için kullanıldı (0,1)
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#epochs=eğitim verilerinin ağa gösterilme sayısı
#batch_size; girdilerin tam katı olmalı gruplandırma sayısını verir
classifier.fit(X_train,Y_train,batch_size=5,epochs=100)

#Modeli test et
test_loss,test_accuracy=classifier.evaluate(X_test,Y_test)

#Test verisi ile modeli test etme
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
from sklearn.metrics import confusion_matrix
com=confusion_matrix(Y_test,Y_pred)
print(com)

from sklearn.metrics import precision_score, recall_score, f1_score,  accuracy_score
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(com, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(com.shape[0]):
    for j in range(com.shape[1]):
        ax.text(x=j, y=i,s=com[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Accuracy: %.3f' % accuracy_score(Y_test,Y_pred))
print('Precision: %.3f' % precision_score(Y_test, Y_pred))
print('Recall: %.3f' % recall_score(Y_test, Y_pred))
print('F1 Score: %.3f' % f1_score(Y_test, Y_pred))
