# 交接檔欄位說明（outputs/predictions.csv）

研究組產生、平台讀取。由 `python -m src.export` 產生，一列一位客戶，依風險分數由高到低排序。
**欄位名稱不要改**，要新增欄位可以，改名或刪除會讓平台壞掉，請先跟組長討論。

| 欄位 | 說明 | 範例 |
|---|---|---|
| customer_id | 客戶 ID（UCI 原始 ID） | 1957 |
| risk_score | 模型預測的違約機率，0 到 1 | 0.9569 |
| risk_rank | 風險排名，1 為最高 | 1 |
| in_list | 是否列入名單（risk_score ≥ 門檻） | 1 |
| sex / age / age_group / education / marriage | 族群資訊，供公平性監控 | 女 / 42 / 40-49 歲 / 研究所 / 已婚 |
| limit_bal | 信用額度 | 220000 |
| bill_amt_latest | 最近一期（9 月）帳單金額 | 2500 |
| actual_default | 實際是否違約（歷史回測才有） | 1 |
| data_split | 資料來源，目前固定為 test | test |
| factor_1～3 | 推高風險的前 3 個原因（TreeSHAP），已轉成白話 | 9 月延遲繳款 2 個月 |

另外輸出：
- `outputs/model.joblib`：訓練好的模型與特徵欄位
- `outputs/meta.json`：產生時間、門檻、模型參數、測試集表現
