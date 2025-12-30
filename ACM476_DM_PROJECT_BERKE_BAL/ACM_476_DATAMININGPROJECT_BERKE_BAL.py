# ======================================================
# ADIM 1 ve 2 BİRLEŞİK: Yükleme ve Temizleme
# ======================================================
import pandas as pd
import numpy as np
import io
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# 1. DOSYAYI YÜKLEME
print("Lütfen 'healthcare-dataset-stroke-data.csv' dosyasını tekrar seç:")
uploaded = files.upload()
dosya_ismi = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[dosya_ismi]))

# 2. VERİ TEMİZLEME (Data Cleaning)
# ID'yi siliyoruz
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# BMI sütunundaki hataları düzeltip boşlukları ortalama ile dolduruyoruz
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

print("\nHarika! Veri yüklendi, ID silindi ve eksik BMI değerleri dolduruldu.")
print("Verinin son hali:")
display(df.head())




# --- 2. Detaylı Veri Keşfi (EDA) ---

# Grafik Ayarları
plt.figure(figsize=(18, 6))

# Grafik 1: İnme Durumunun Dağılımı (Pasta Grafiği)
plt.subplot(1, 3, 1)
df['stroke'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], explode=[0, 0.1], shadow=True)
plt.title("İnme Geçirme Oranı (Dengesiz Veri Seti)")
plt.ylabel('')

# Grafik 2: Yaş ve BMI İlişkisinin İnme ile Dağılımı (Scatter Plot)
plt.subplot(1, 3, 2)
sns.scatterplot(x='age', y='bmi', hue='stroke', data=df, alpha=0.6, palette='coolwarm')
plt.title("Yaş ve BMI İlişkisi (Kırmızı: İnme Var)")

# Grafik 3: Korelasyon Isı Haritası (Sadece Önemli İlişkiler)
plt.subplot(1, 3, 3)
corr = df.corr()
# Sadece stroke ile ilişkisi 0.05'ten büyük olanları gösterelim (Gürültüyü azaltmak için)
target_corr = corr[['stroke']].sort_values(by='stroke', ascending=False)
sns.heatmap(target_corr, annot=True, cmap='Reds', linewidths=2)
plt.title("İnme ile En Yüksek İlişkili Özellikler")

plt.tight_layout()
plt.show()





# --- 3. PCA (Temel Bileşen Analizi) Görselleştirmesi ---
from sklearn.decomposition import PCA

# Veriyi Ölçekle (PCA için şarttır)
scaler = StandardScaler()
X = df.drop('stroke', axis=1)
X_scaled = scaler.fit_transform(X)

# 2 Bileşene İndir
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df['stroke'], palette='seismic', alpha=0.7)
plt.title(f"Veri Setinin 2 Boyutlu PCA Gösterimi\n(Varyansın %{sum(pca.explained_variance_ratio_)*100:.1f}'i açıklandı)")
plt.xlabel("Bileşen 1")
plt.ylabel("Bileşen 2")
plt.show()

print("YORUM: Kırmızı noktalar (İnme vakaları) grafiğin sağ tarafında yoğunlaşıyor mu? Eğer öyleyse, modelimiz bunları kolay ayırt edebilir demektir.")







# --- 4. Gelişmiş Modelleme ve Raporlama ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Eğitim/Test
X_train, X_test, y_train, y_test = train_test_split(X, df['stroke'], test_size=0.2, random_state=42)

# Model: Random Forest (Daha güçlü parametrelerle)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sonuçları Görselleştir
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
plt.title("Confusion Matrix (Hata Matrisi)")
plt.ylabel("Gerçek Durum")
plt.xlabel("Tahmin Edilen")
plt.show()

# Özellik Önemi (Feature Importance) - Hangisi Etkili?
importance = pd.DataFrame({'Özellik': X.columns, 'Önem': model.feature_importances_})
importance = importance.sort_values(by='Önem', ascending=False).head(5)

print("\n--- İNMEYİ ETKİLEYEN EN KRİTİK 5 FAKTÖR ---")
plt.figure(figsize=(10, 4))
sns.barplot(x='Önem', y='Özellik', data=importance, palette='viridis')
plt.title("Modelin Karar Verirken Baktığı En Önemli Kriterler")
plt.show()






# ==========================================================
# ADIM 5: MAKİNE ÖĞRENMESİ MODELİ KURMA VE TEST ETME
# ==========================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. VERİYİ HAZIRLAMA (Encoding)
# Makine "Male", "Yes" gibi yazıları anlamaz. Bunları sayıya çeviriyoruz.
df_model = df.copy() # Orijinal veriyi bozmayalım diye kopyasını alıyoruz
le = LabelEncoder()

# Yazı olan tüm sütunları bul ve sayıya çevir
for col in df_model.select_dtypes(include=['object']).columns:
    df_model[col] = le.fit_transform(df_model[col])

# Girdiler (X) ve Hedef (y) ayrımı
X = df_model.drop('stroke', axis=1) # İnme sütunu hariç hepsi ipucu
y = df_model['stroke']              # Tahmin etmeye çalıştığımız şey

# 2. EĞİTİM VE TEST DİYE AYIRMA
# Verinin %80'i ile ders çalışacak, %20'si ile sınava girecek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Model Eğitiliyor... (Eğitim Verisi: {X_train.shape[0]} kişi, Test Verisi: {X_test.shape[0]} kişi)")

# 3. MODELİ KURMA (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # Modeli eğitiyoruz

# 4. TEST ETME (Sınav Zamanı)
tahminler = model.predict(X_test)

# 5. SONUÇLARI RAPORLAMA
basari_orani = accuracy_score(y_test, tahminler)
print(f"\nModel Doğruluk Oranı (Accuracy): %{basari_orani*100:.2f}")

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, tahminler))

# 6. GÖRSELLEŞTİRME: Confusion Matrix (Hata Matrisi)
# Model nerede hata yaptı?
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, tahminler)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1)
plt.title('Hata Matrisi (Confusion Matrix)\nDoğru ve Yanlış Tahminler')
plt.ylabel('Gerçek Durum')
plt.xlabel('Modelin Tahmini')
plt.show()

# 7. GÖRSELLEŞTİRME: Özellik Önemi (Feature Importance)
# Model karar verirken en çok neye baktı?
onem_dereceleri = pd.DataFrame({'Özellik': X.columns, 'Önem': model.feature_importances_})
onem_dereceleri = onem_dereceleri.sort_values(by='Önem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Önem', y='Özellik', data=onem_dereceleri, palette='viridis')
plt.title('İnmeyi Belirleyen En Kritik Faktörler')
plt.show()





# ==========================================================
# ADIM 6: İLERİ SEVİYE ANALİZ VE SİMÜLASYON
# ==========================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# --- A. HASTA SEGMENTASYONU (K-MEANS CLUSTERING) ---
# Amacımız: Hastaları benzerliklerine göre gruplayıp "Risk Profilleri" çıkarmak.
print("--- K-Means ile Hasta Gruplama Analizi ---")

# Sadece sayısal verileri alıp ölçekleyelim (Standartlaştırma)
# Çünkü yaş 80 iken şeker 200 olabilir, bu fark makineyi şaşırtmasın diye eşitliyoruz.
scaler = StandardScaler()
cols_to_cluster = ['age', 'avg_glucose_level', 'bmi']
X_scaled = scaler.fit_transform(df[cols_to_cluster])

# Hastaları 3 ana gruba ayıralım
kmeans = KMeans(n_clusters=3, random_state=42)
df['Risk_Grubu'] = kmeans.fit_predict(X_scaled)

# Her grubun özelliklerine bakalım
grup_ozellikleri = df.groupby('Risk_Grubu')[['age', 'avg_glucose_level', 'bmi', 'stroke']].mean()
print("\nBilgisayarın Oluşturduğu 3 Hasta Profili:")
display(grup_ozellikleri)
print("\nYORUM: Yukarıdaki tabloda 'stroke' oranı en yüksek olan grup, en riskli hasta profilidir.")

# --- B. YENİ HASTA SİMÜLASYONU (PREDICTION) ---
# Eğittiğimiz 'model'i kullanarak sanal hastalar için tahmin yapalım.
print("\n--- Yeni Hasta Risk Hesaplama Simülasyonu ---")

def risk_hesapla(yas, seker, bmi, tansiyon=0, kalp=0, sigara=0):
    # Modelin beklediği formatta veri oluşturuyoruz
    # Not: Diğer sosyal verileri (evlilik, iş) ortalama/yaygın değerler alıyoruz
    yeni_veri = pd.DataFrame({
        'gender': [1], # Erkek varsaydık
        'age': [yas],
        'hypertension': [tansiyon],
        'heart_disease': [kalp],
        'ever_married': [1], # Evli varsaydık
        'work_type': [2],    # Özel sektör varsaydık
        'Residence_type': [1], # Şehir varsaydık
        'avg_glucose_level': [seker],
        'bmi': [bmi],
        'smoking_status': [sigara],
        'Risk_Grubu': [0] # Teknik bir detay, etkisiz
    })

    # Model eğitimi sırasında 'Risk_Grubu' sütunu yoktu, onu çıkaralım
    if 'Risk_Grubu' in yeni_veri.columns:
        yeni_veri = yeni_veri.drop('Risk_Grubu', axis=1)

    # Tahmin yapıyoruz
    # predict_proba bize % kaç ihtimalle inme olacağını verir
    olasilik = model.predict_proba(yeni_veri)[0][1]
    return olasilik

# ÖRNEK SENARYOLAR
# Senaryo 1: 25 yaşında, sağlıklı bir birey
risk1 = risk_hesapla(yas=25, seker=85, bmi=22)
print(f"Senaryo 1 (25 Yaş, Sağlıklı): İnme Riski %{risk1*100:.2f}")

# Senaryo 2: 72 yaşında, şekeri yüksek, tansiyon hastası bir birey
risk2 = risk_hesapla(yas=72, seker=210, bmi=32, tansiyon=1, kalp=0)
print(f"Senaryo 2 (72 Yaş, Yüksek Şeker/Tansiyon): İnme Riski %{risk2*100:.2f}")

if risk2 > risk1:
    print("\nSONUÇ: Modelimiz, yaş ve kronik hastalıkların riski ciddi oranda artırdığını başarıyla öğrenmiştir.")
    
    
    
    
    
    
# ==========================================================
# ADIM 7: İNTERAKTİF İNME RİSKİ HESAPLAYICI (Kullanıcı Girişli)
# ==========================================================
import pandas as pd

print("--- İNME RİSKİ HESAPLAMA SİSTEMİNE HOŞ GELDİNİZ ---")
print("Lütfen analiz için aşağıdaki değerleri giriniz:\n")

# 1. KULLANICIDAN VERİ ALMA
try:
    # Sayısal değerleri istiyoruz
    yas = float(input("1. Yaşınız kaç? (Örn: 25): "))
    seker = float(input("2. Ortalama Şeker Düzeyiniz? (Örn: 100): "))
    bmi = float(input("3. BMI (Vücut Kitle İndeksi) değeriniz? (Örn: 24.5): "))

    # Evet/Hayır sorularını 1 ve 0 olarak alalım
    tansiyon = int(input("4. Hipertansiyon (Tansiyon) var mı? (Evet:1, Hayır:0): "))
    kalp = int(input("5. Kalp Hastalığı var mı? (Evet:1, Hayır:0): "))

    print("\nVeriler alınıyor ve yapay zeka modeline soruluyor...")

    # 2. MODEL İÇİN VERİ HAZIRLAMA
    # Model eğitimi sırasında kullanılan tüm sütunları oluşturmamız lazım.
    # Diğer detayları (evlilik, iş vb.) 'ortalama' veya 'yaygın' değerler olarak sabitliyoruz.
    # (Çünkü en kritik olanlar yukarıda sorduklarımız)

    yeni_hasta_verisi = pd.DataFrame({
        'gender': [1],           # Varsayılan: Erkek (1) veya Kadın (0) - Etkisi düşüktür
        'age': [yas],            # GİRDİĞİNİZ DEĞER
        'hypertension': [tansiyon], # GİRDİĞİNİZ DEĞER
        'heart_disease': [kalp],    # GİRDİĞİNİZ DEĞER
        'ever_married': [1],     # Varsayılan: Evet
        'work_type': [2],        # Varsayılan: Özel Sektör
        'Residence_type': [1],   # Varsayılan: Şehir
        'avg_glucose_level': [seker], # GİRDİĞİNİZ DEĞER
        'bmi': [bmi],            # GİRDİĞİNİZ DEĞER
        'smoking_status': [1]    # Varsayılan: İçmiyor/Bırakmış
    })

    # Modelin beklediği sütun sırasını garantiye alalım (Hata çıkmasın diye)
    # X değişkeni 5. Adım'dan hafızada kalmıştı
    if 'Risk_Grubu' in yeni_hasta_verisi.columns:
        yeni_hasta_verisi = yeni_hasta_verisi.drop('Risk_Grubu', axis=1)

    # Sütun sırasını eşitleme
    # Eğer önceki adımlarda X tanımlıysa onun sütunlarını kullan
    # Değilse manuel sıralama yapmayalım, DataFrame genelde doğru oluşturur.
    try:
        yeni_hasta_verisi = yeni_hasta_verisi[X.columns]
    except:
        pass # X hafızada yoksa devam et (Genelde sorun olmaz)

    # 3. TAHMİN YAPMA
    # predict_proba: Bize [İnmeYok_Oranı, İnmeVar_Oranı] verir. Biz 1. indeksi (Var) alıyoruz.
    risk_orani = model.predict_proba(yeni_hasta_verisi)[0][1]

    # 4. SONUCU YAZDIRMA
    print("-" * 40)
    print(f"SONUÇ: Girilen verilere göre İnme Riski: %{risk_orani*100:.2f}")

    if risk_orani > 0.50:
        print("⚠️ DİKKAT: Yüksek risk grubu! Bir doktora görünmeniz önerilir.")
    elif risk_orani > 0.20:
        print("⚠️ UYARI: Orta seviye risk. Sağlıklı beslenmeye dikkat edin.")
    else:
        print("✅ DURUM: Düşük risk. Gayet sağlıklı görünüyorsunuz.")
    print("-" * 40)

except ValueError:
    print("\n❌ HATA: Lütfen sadece sayısal değerler giriniz (Örn: 'Var' yerine 1 yazınız).")
    
    
    
    
    

# ==========================================================
# EK ANALİZ: SOSYAL VE DEMOGRAFİK DEĞİŞKENLERİN ETKİSİ
# ==========================================================
# Burada sayısal olmayan (Kategorik) verileri inceliyoruz.

import seaborn as sns
import matplotlib.pyplot as plt

# Grafiklerin sığacağı bir alan açalım (2 satır, 2 sütun)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Grafik: Evlilik Durumu ve İnme
# (Evlilerde mi bekarlarda mı risk fazla?)
sns.countplot(data=df, x='ever_married', hue='stroke', palette='Set1', ax=axes[0, 0])
axes[0, 0].set_title('Evlilik Durumuna Göre İnme Sayıları')
axes[0, 0].set_xlabel('Hiç Evlendi mi?')
axes[0, 0].set_ylabel('Kişi Sayısı')

# 2. Grafik: Çalışma Tipi ve İnme
# (Hangi meslek grubu risk altında?)
sns.countplot(data=df, x='work_type', hue='stroke', palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('Çalışma Tipine Göre İnme Riski')
axes[0, 1].set_xlabel('Çalışma Şekli')
axes[0, 1].tick_params(axis='x', rotation=45) # Yazılar sığsın diye eğiyoruz

# 3. Grafik: İkamet Yeri (Kırsal vs Şehir)
# (Köy havası alanlar daha mı sağlıklı?)
sns.countplot(data=df, x='Residence_type', hue='stroke', palette='Set3', ax=axes[1, 0])
axes[1, 0].set_title('İkamet Yerine Göre İnme Dağılımı (Kırsal vs Şehir)')
axes[1, 0].set_xlabel('Yaşadığı Yer')

# 4. Grafik: Sigara Kullanımı
# (Sigaranın etkisi ne kadar?)
sns.countplot(data=df, x='smoking_status', hue='stroke', palette='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Sigara Kullanımına Göre İnme')
axes[1, 1].set_xlabel('Sigara Durumu')

plt.tight_layout()
plt.show()

# --- YORUM İÇİN ORANLARI YAZDIRALIM ---
print("\n--- İSTATİSTİKSEL ORANLAR ---")
print("Evlilerde İnme Oranı: ", df[df['ever_married'] == 1]['stroke'].mean())
print("Bekarlarda İnme Oranı: ", df[df['ever_married'] == 0]['stroke'].mean())
print("-" * 30)
print("Şehirde Yaşayanlarda Risk: ", df[df['Residence_type'] == 1]['stroke'].mean())
print("Kırsalda Yaşayanlarda Risk: ", df[df['Residence_type'] == 0]['stroke'].mean())
    