from newspaper import Article, build
from urllib.parse import urlparse
from datetime import datetime, timedelta
import json

def crawl_last_7_days(urls, output_file='articles_last_7_days.json'):
    result = {}
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)

    for url in urls:
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            print(f"\n📡 Memproses situs: {domain}")
            result[domain] = []

            paper = build(url, memoize_articles=False)
            print(f"  Ditemukan {len(paper.articles)} artikel di halaman utama")

            count_saved = 0

            for idx, article in enumerate(paper.articles):
                try:
                    article.download()
                    article.parse()

                    # Jika tanggal publikasi tersedia dan di dalam 7 hari terakhir
                    if article.publish_date:
                        if article.publish_date < seven_days_ago or article.publish_date > now:
                            print(f"    ❌ Lewati artikel {idx+1}: di luar rentang waktu ({article.publish_date})")
                            continue

                    result[domain].append({
                        "title": article.title,
                        "publish_date": str(article.publish_date),
                        "text": article.text[:500]  # potong untuk ringkasan
                    })
                    print(f"    ✅ Artikel {idx+1} disimpan: {article.title}")
                    count_saved += 1

                except Exception as e:
                    print(f"    ⚠️  Gagal memproses artikel {idx+1}: {e}")

            print(f"  ✅ Total artikel disimpan dari {domain}: {count_saved}")

        except Exception as e:
            print(f"❌ Gagal membangun daftar artikel dari {url}: {e}")

    # Simpan ke file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Semua data disimpan ke '{output_file}'")

# Contoh daftar situs berita
urls = [
    'https://www.kompas.com',
    'https://www.cnnindonesia.com',
    'https://www.detik.com'
]

crawl_last_7_days(urls)
