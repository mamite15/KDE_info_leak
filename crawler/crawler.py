import os
import time
import subprocess
from selenium import webdriver
from urllib.parse import urlparse
import crawler.parse as parse

# *** URLリストをファイルから取得する関数 ***
def get_url_list(input_path):
    # ファイルのオープン
    targetFile = open(input_path)
    # 行ごとのURLのリスト
    url_list = targetFile.readlines()
    # ファイルのクローズ
    targetFile.close()
    return url_list

# *** ブラウジングを実行する関数 ***
def browsing_urls(url_list):
    options = webdriver.ChromeOptions()

    #ヘッドレス環境(ブラウザを表示しない)
    options.add_argument('--headless')
    # ブラウザを最大化
    options.add_argument("--start-maximized")
    # シークレットモードでの実行
    #options.add_argument('--incognito')
    #キャッシュを無効
    #options.add_argument('--disable-cache')
    #options.add_argument("--disk-cache-size=0")
    #options.add_argument("--clear-cache")

    # 「Chromeは自動テストソフトウェアによって制御されています。」を消すためのオプションの指定
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    print("\n===== start =====")
    #トレース数カウント
    trace_cnt=0
    #各サイトに1000回訪問
    for num in range(1000):
        #1行ずつ読み込んで各サイトにアクセス
        for url in url_list:
            trace_cnt+=1
            #chromeドライバ起動
            driver = webdriver.Chrome(options=options)
            #ドメイン名をurlから取り出す
            domain=urlparse(url).netloc
            print(domain + "へのアクセス" + str(num+1) + "回目")
            #スクリーンショットのファイル名指定
            file="image/" + domain + "_" + str(num+1) + ".png"
            FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            #pcapファイル名の指定
            pcapfile="pcap/" + domain + "_" + str(num+1) + ".pcap"
            #tcpdumpの起動
            subprocess.Popen(["sudo","tcpdump","-i","enxcce1d50d3d69","-w", pcapfile])
            #tcpdump起動を一応待つ
            time.sleep(2)
            #URLにアクセス
            driver.get(url)
            #完全にトレースできるために30秒待つ
            time.sleep(30)
            #スクリーンショット　撮影
            driver.save_screenshot(FILEPATH)
            #tcpdump停止
            subprocess.run(["sudo","pkill","tcpdump"])
            print()
            #不正アクセスと見做されないように30秒待つ
            time.sleep(30)
            #chromeドライバ閉じる
            driver.close()
    print("===== end =====\n")
    # 終了
    print("クロール完了\n")
    return trace_cnt

def main():
    # URL一覧ファイルの受付
    input_path = "test_url_list2.txt"
    # URLリストをファイルから取得
    url_list = get_url_list(input_path)
    # ブラウジング
    trace_cnt=browsing_urls(url_list)
    #パケット解析
    parse.parse_main(trace_cnt)

# main関数の実行
main()