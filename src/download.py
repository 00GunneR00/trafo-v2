"""ETTh1 verisini kaynağından indirir (data/raw/ETTh1.csv).

Veri depoya konmaz; lisansı ve sürümü kaynağında kalsın: https://github.com/zhouhaoyi/ETDataset
"""
import urllib.request

from data import RAW

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

if __name__ == "__main__":
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, RAW)
    print(f"İndirildi: {RAW}")
