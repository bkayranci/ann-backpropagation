from numpy import exp, array, random, dot

agirliklar = 2 * random.random((3, 1)) - 1

def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_turevi(x):
    return x * (1 - x)


def egit(egitim_girdileri, egitim_ciktilari, dongu_sayisi):
    for _ in range(dongu_sayisi):
        # Pass the training set through our neural network (a single neuron).
        output = karar_ver(egitim_girdileri)

        # hata hesapla
        hata = egitim_ciktilari - output

        # sifir olan girisler agirlik degisimine neden olmaz
        ayarlanacak_agirlik = dot(egitim_girdileri.T, hata * sigmoid_turevi(output))

        # agirliklari yeniden belirle
        global agirliklar
        agirliklar += ayarlanacak_agirlik

# karar ver
def karar_ver(inputs):
    return sigmoid(dot(inputs, agirliklar))




print("Rastgele agirliklar: ")
print(agirliklar)
egitim_girdileri = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
egitim_ciktilari = array([[0, 1, 1, 0]]).T

# egitim
egit(egitim_girdileri, egitim_ciktilari, 8000)

print("Egitim sonrasi agirliklar: ")
print(agirliklar)

# yeni bir deger ile test et
print("Karar: [1, 1, 0] -> ?: ")
print(karar_ver(array([1, 1, 0])))

# kullanicidan deger alarak karar ver
# while True:
#     test = input("Karar verilmesi icin 3 adet degeri aralarinda bosluk olacak sekilde girin: ")
#     try:
#         data = [int(i) for i in test.split(' ')]
#     except Exception as ex:
#         print("[HATA] 3 adet degeri bosluklarla girmediniz!")
#         continue
#     if len(data) == 3 and all([isinstance(item, int) for item in data]):
#         print("Karar: [{}, {}, {}] -> ?: ".format(data[0], data[1], data[2]))
#         print(karar_ver(array(data)))
#         break
#     else:
#         print("[HATA] 3 adet degeri bosluklarla girmediniz!")
