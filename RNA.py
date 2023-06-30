import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

entrada=input("Deseja criar um modelo novo ou carregar um modelo? (1/2) \n")

#Leitura dos dados
df1 = pd.read_csv('Entrada_normalizada.csv') # Usar valores normalizados para sigmoid
df2 = pd.read_csv('Saida_normalizada.csv')

#Dividindo os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.1)
if(entrada=="1"):
    #Criando o modelo, adicionando as camadas de neuronios e definindo a função de ativação de cada uma
    model = Sequential()
    model.add(Dense(32, input_shape=(1,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1,))

    #Compilando o modelo
    model.compile(Adam(learning_rate=0.001), 'mse')

    #Treinando o modelo
    history = model.fit(X_train, y_train, epochs = 6000,validation_split=0.2, verbose = 1)

    #Plotando no gráfico o Loss e a Validação
    plt.plot(history.history['loss'],'r', label = 'Loss')
    plt.plot(history.history['val_loss'],'b', label = 'Validacao')
    plt.show()
elif(entrada=="2"):
    #Carregando um modelo já treino
    arq=input("Digite o nome do Modelo que deseja carregar: ")
    model = tf.keras.models.load_model(arq)
    model.summary()
if entrada =='1' or entrada =='2':
    #Pegando o y previsto com valores de teste para utilizar no cálculo do score
    y_pred = model.predict(X_test)
    #print("Score: ", r2_score(y_test,y_pred))

    #Pegando o y previsto com todos os valores para plotar os gráficos
    y_pred = model.predict(df1) 

    #Plotando o gráfico dos Valores previstos e lidos
    plt.plot(df1,y_pred*802,'b', label = 'ValorPrevisto') # Multiplicar por 802 para sigmoid
    plt.plot(df1,df2*802,'r', label='ValorRead') # Multiplicar por 802 para sigmoid
    plt.show()

    #Salvando ou não o modelo
    cond = 'N'
    if(entrada=='1'):
        cond=input("Deseja salvar? Y/N \n")
    if(cond == "Y"):
        Nome = input("Digite o nome que deseja salvar: ")
        model.save(Nome) 