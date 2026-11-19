# L2 與 L3 正式評估入口

本文件說明 C9b-2 的介面，不代表已核准 G2b 或已執行真實評估。正式執行需組長另外授權，本卡僅使用人工資料與 Mock 測試。

## 命令與前置條件

```text
python -m src.final_eval <model_freeze_id>
python -m src.stability_eval <model_freeze_id>
```

兩者先以 `src/_evaluation.py::validate_release` 驗證整份核准清單、G2a 規則、核准者與時間、必要 entry、所有模型與政策來源、校準器、草稿及候選報告，再透過 protocol 取得 L2 或 L3 資料。任一項不符就停止，不呼叫資料入口。這些命令不建立或修改凍結紀錄，也不開封 L4。

L2 使用凍結模型，每個唯一模型只產生一次分數，按客戶 ID 對齊，政策不重新搜尋。只有 platform-c 在本機 outputs 下保存交接檔與 SHAP 風險因素，重載模型核對全部 6,000 人後才完成。其他結果都是彙總表。app/data.py 的平台 L1-H 限制維持原樣。

L3 的 seed 清單、五折設定、模型參數、特徵與政策程序來自凍結清單及規則。每個 seed 的唯一模型都重新產生 OOF 與完整 fit，三種政策共用該訓練來源。S 與 N 固定，C、F3b、E 與校準後 C 只在該 seed 的訓練 OOF 選定。F7 先交叉擬合 OOF 選政策，再保存供該 seed 評估使用的全 OOF 校準器。模型本身的 random_state 不跟著外層切分 seed 改變。

## 識別與來源證據

- 每個 entry 與政策各有自己的 run_id。模型來源以 model_source_run_id 與 retraining.fit_run_id 追溯，並記錄訓練 IDs、OOF 分數、折分配、模型設定雜湊及實際 fit_seconds。
- L3 只有在新的 model_id、L3 模型紀錄與獨立 fit 證據一致時，才接受與凍結模型相同的真實 hash。L1／L2 工件不能直接綁進 L3。同模型的其他政策必須引用已完成的 L3 fit 來源。
- training_timing.csv 保留 fit、OOF、政策、校準器、工件與 wall 時間、含時區的開始結束時間、硬體、執行緒與版本。政策共用 fit 時間來源，不能把三政策相加當三次訓練。外層 split_seed 與 cv_seed 分開記錄。校準器擬合、交叉擬合與政策時間分開量測，不將校準耗時算進 policy_seconds。
- summary_id 屬於整批彙總，不冒充模型 run。manifest 綁定凍結檔雜湊、各 task 狀態、run_id、模型及政策雜湊、run.json 與結果檔 SHA256。

## 輸出

`results/<entry_run_id>/` 保存 overall、groups、gaps、gap_ci、截取前的 raw_policy_*、classification_report、confusion_matrix、table_4_1_row、training_timing 與 run.json。RQ3 只輸出 rq3_lead_time、rq3_coverage、時間及 run.json，不填假政策。F7 另有校準摘要、分箱、曲線與前 N 名單重疊率。

表 4-1 的 Accuracy 從同一份最終名單的 TN、FP、FN、TP 計算，沒有修改 evaluate 的固定欄位。表 4-3 保留兩類及 macro，零分母為空值並附 *_undefined。任一類對應指標未定義時，macro 也為空值。macro 的 support 是總人數。平台 C 的混淆矩陣圖以原始人數標記，真實與預測類別順序均為 0、1，標題含完整來源識別。

L2 的 RQ1 另有 M1 與 M1-NS 各一個比較 run，保存與 06 雙 run 格式相同的 fairness_overall.csv、fairness_groups.csv、fairness_gaps_point.csv、fairness_gaps.csv。三種名稱固定為共同門檻0.5、固定名單前20%、正式政策。M1-NS 的成對列保留 m1_run_id、m1_model_id 及 B-A 方向，可直接以 `app.data.load_fairness_comparison` 讀取。自助法的次數與 seed 由凍結的 D8 讀取。

`results/<summary_id>/` 包含 evaluation_manifest.json、source_runs.csv 與該層級報表。L2 有 table_4_1_l2.csv、table_4_3_l2.csv、rq3_summary.csv。L3 有 stability_by_seed.csv、stability_summary.csv、table_4_1_l3.csv、rq3_summary.csv、rq3_coverage_summary.csv。表 4-1 同時保留完整政策表與 *_main.csv 正式 C 政策主列，分類來源列附 is_primary。L3 按 entry、比較、模型設定、政策種類與規則分組，不按每個 seed 的 policy_id 拆散。彙總為 mean、std（ddof=1）、min、max、n_valid，少一個 seed 就不發布完成的十次彙總。

## 失敗與重試

錯誤會停止整批，當前 run 標記 failed，完成的 run 保持不變，尚未開始的 task 保留 pending。失敗 manifest 不會變成正式完成結果。

```text
python -m src.final_eval <model_freeze_id> --resume <失敗的summary_id>
python -m src.stability_eval <model_freeze_id> --resume <失敗的summary_id>
```

重試必須是同層級、同一份凍結檔與相同 task 清單。程式先驗證既有完成 run 與結果檔的雜湊，再建立新的 summary_id，引用已完成的 run，只補跑失敗或未執行項目。L3 可重載已完成的該 seed 訓練來源，不重新 fit 已完成的唯一模型或已完成的校準器，校準器另以 calibration_source_run_id 追溯。舊 manifest、完成 run 與凍結檔都不覆寫。如果修正涉及研究設定或凍結內容，須先由組長決定是否建立新 freeze，不能沿用舊 ID 修改內容。

L3 是切分穩定性分析，十次資料互有重疊，不是十次獨立外部驗證。任何 L2、L3 或 L4 結果都不得回饋調整研究設計。

## 彙總識別與差距快取

彙總與 F3 讀回 CSV 時，model_key 與各種 *_id、雜湊等識別欄位保留原始字串，例如 05 不轉成數字 5。數值指標與未定義空值維持原有解析。

差距區間快取只存在各自的 Batch 物件，不寫入檔案或跨批次共用。快取鍵包含 y、pred、pred_b、客戶 ID 與順序、所有分組名稱／內容／順序、n_boot 與 seed 的內容雜湊。相同輸入重用原結果的副本，內容雜湊不同就重新計算，不改變 bootstrap 算法或數值。
