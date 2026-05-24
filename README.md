\# 🚗 車両データ分析アシスタント



自然言語で車両データに質問できるAIアシスタントです。



\## デモ



https://car-app-qkqjbrd2gp65vdcegkxfep.streamlit.app



\## 機能



\- 自然言語で車両データに質問できる

\- 集計系の質問（何台？平均価格は？）に正確に回答

\- 検索系の質問（ディーゼル車は？）にRAGで回答

\- 燃料タイプ別・年式別のグラフ表示



\## 使用技術



| 技術 | 用途 |

|---|---|

| Python | メイン言語 |

| Streamlit | Web UI |

| Claude API | 自然言語処理 |

| FAISS | ベクトル検索 |

| sentence-transformers | テキストのベクトル化 |

| pandas | データ分析 |



\## アーキテクチャ

ユーザーの質問

　↓

質問タイプ判定（集計系 or 検索系）

　↓

集計系: pandasで正確に計算

検索系: FAISSでベクトル検索

　↓

Claude APIで日本語回答生成



\## セットアップ



```bash

git clone https://github.com/akinorifujii-creator/car-app.git

cd car-app

pip install -r requirements.txt

streamlit run app.py

```



\## データ



\[Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)



\- 301台の中古車データ

\- インド市場の車両情報



\## 作者



Akinori Fujii

