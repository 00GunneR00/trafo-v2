# Kararlar

Projede verilen her kararın gerekçesi. Mülakatta "neden böyle yaptın?" sorusunun cevabı burada.

## 1. Veri: ETTh1

- Kaynak: [ETDataset](https://github.com/zhouhaoyi/ETDataset), Çin'de gerçek bir trafo, saatlik, Tem 2016 – Haz 2018.
- Sütunlar: `OT` yağ sıcaklığı (hedef), `HUFL/HULL/MUFL/MULL/LUFL/LULL` yüksek/orta/düşük seviye faydalı (UFL) ve faydasız (ULL) yük.
- Eksik değer yok, adım her zaman 1 saat (`src/explore.py`).
- **Ortam sıcaklığı yok.** Yağ sıcaklığını en çok etkileyen etkenlerden biri eksik; tahmin hassasiyetini sınırlar.
- **Arıza etiketi yok.** Bu yüzden "arıza tahmini" iddia edilmez (bkz. 4).

## 2. Bölme: zamana göre, 12 / 4 / 4 ay

- ETT makalesindeki (Informer, Zhou ve ark. 2021) standart bölme; sonuçlar literatürle kıyaslanabilir.
- **Rastgele bölme yapılmaz:** komşu saatler neredeyse aynıdır (1 saatlik otokorelasyon 0,993). Rastgele bölünce test saatinin bir saat öncesi ve sonrası eğitimde olur, model "geleceği" görmüş gibi iyi görünür. Gerçek kullanımda model hep geçmişle eğitilip gelecekte çalışır; değerlendirme de böyle olmalı.
- **Ölçekleme yalnızca train'e göre ayarlanır.** Ortalama/standart sapma test dahil hesaplanırsa test verisinin bilgisi eğitime sızar.

## 3. Problem: OT'yi 1, 6 ve 24 saat ileriye aralıklı tahmin

- Girdi: son 168 saat (7 gün) × 7 ölçüm + saat/gün döngüsü özellikleri. 7 gün, haftalık döngüyü (otokorelasyon 168 saatte 0,853) görmeye yetecek kadar.
- **Dağılım kayması:** train ortalaması 17,1 °C, test (kış) 4,9 °C. Seviyeyi doğrudan öğrenen model yanlış seviyeye çekilir.
  Bu yüzden hedef **değişim**: `OT(t+h) − OT(t)`. Seviye ile değişim hedefi deneyde kıyaslanır.
- **Aralık tahmini:** her ufuk için %5, %50, %95 yüzdelikleri (quantile). Ortanca tahmin, ikisinin arası %90 aralık.
  Kayıp fonksiyonu: pinball (quantile) loss.
- Ölçütler: MAE, RMSE (ortanca üzerinden, °C), aralık kapsaması (hedef %90) ve ortalama aralık genişliği.
- Kıyas çıtaları: son değer, dünün aynı saati, doğrusal regresyon, gradient boosting. LSTM bunları geçmiyorsa sonuç yine raporlanır.

### Deneyde çıkanlar

- Gradient boosting, mutlak sıcaklık gecikmeleriyle testte çöktü (24 sa MAE 3,39 °C, kapsama %69). Gecikmeler
  şu anki değere göre farka çevrilince 1,68 °C ve %92'ye geldi. Aynı fikir LSTM girdisine de uygulandı (`make_relative`).
- Seviye hedefli LSTM testte 3,31 °C; değişim hedefli 1,57 °C. Karar 3 doğrulandı.
- LSTM 3. epoch civarında ezberlemeye başlıyor; erken durdurma en iyi val ağırlığına dönüyor.
- 2×64 ve 2×32 val'de aynı; küçük olan seçildi (daha az parametre, aynı başarı).
- Nokta tahmininde hiçbir model "son değer"i testte güvenilir biçimde geçmiyor. Raporlandı, gizlenmedi.

### Kalibrasyon

- Ham aralıklar fazla geniş (val %93-94). Conformalized quantile regression ile val'de %90'a oturtuldu.
- Alarm için ayrı, tek yönlü %99 üst sınır kalibre edildi.

## 4. Alarm: beklenmeyen ısınma (anomali)

- Mantık: soğutma arızasında yük aynıyken yağ beklenenden fazla ısınır.
- İlk kural: gerçek sıcaklık kalibre %99 üst sınırını art arda N saat aşarsa alarm. Sonuç zayıf:
  3 °C kademeli arızada %20. Nedeni: 24 saatlik aralık geniş (~±4,5 °C), tek saat nadiren aşıyor.
- İkinci kural: **CUSUM** (Page, 1954). Ölçekli artıkları biriktirir; küçük ama süren sapmayı yakalar.
  Aynı yanlış alarm şartında 3 °C kademeli arızada %75. API bu kuralı kullanıyor.
- Etiket olmadığı için ölçüm **yapay arıza** ile yapılır: test verisine farklı büyüklükte (1–5 °C) kademeli ısınmalar eklenir.
  Ölçülenler: yakalama oranı, yakalama gecikmesi, temiz veride haftalık yanlış alarm.
- N ve kullanılan ufuk **doğrulama** verisinde seçilir, sonuç **test**te raporlanır.
- Sınır: sonuçlar tasarlanan arıza tipine bağlıdır; gerçek arıza yakalama oranı olarak sunulmaz.
