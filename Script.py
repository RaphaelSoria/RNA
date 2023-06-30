import tensorflow as tf
arq=input("Digite o nome do Modelo que deseja carregar: ")
model = tf.keras.models.load_model(arq)
X = input("Digite o valor que deseja testar: ")
if arq == "TesteSigmoid":
    y = model.predict([float(X)/60])
    print("Valor previsto: ", y*802)
else: 
    y = model.predict([float(X)])
    print("Valor previsto: ", y)
    