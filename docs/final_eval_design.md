# C9：G2b 模型凍結與正式執行設計（待組長核准）

本文件只提出設計，不代表下列程式、模型、凍結清單或正式結果已存在。本次未讀取測試集、未執行實驗，也不建立任何凍結紀錄。核對基準為 main `8f6e25d`（2026-11-19）；規格依 `docs/final_spec.md` 第 4.1、4.2、4.5、5、9 節，以及 `docs/frozen_rules.md`。新增 DT0 依本次組長指示，不把它加入 G2b 替換候選。

以下區分「規格已決定」「擬議實作」與「待決定」。本文件的路徑、額外欄位及命令是後續實作提案；第 8 節未核准的研究取捨不可由程式預設補齊。所有數值設定讀取凍結紀錄，不解析 Markdown 或沿用實驗檔中可能過時的常數。

## 1. 現有程式與串接缺口

本次只閱讀程式來源，沒有 import 或執行實驗模組。

| 來源 | main 現況 | C9 後續串接方式／阻擋條件 |
|---|---|---|
| `src/protocol.py` | 已有 `dev_data`、`final_test`、`outer_splits`、`prepare`、折內權重模型、OOF、政策、評估及 run；沒有 `final_eval` | 保留第 4.4 節公開函式名稱與參數；新 orchestrator 組合既有介面，不另寫切分／成本／名單演算法 |
| `src/registry.py` | `frozen_entry` 會核對 L1 來源 run、24,000 人、模型／政策／特徵證據；沒有建立 model_freeze 的命令 | 建置器先產生完整 L1 來源，再核對每筆；不能用手填雜湊或 L1-H run 充數 |
| `experiments/03_repeated_cv.py` | `rq3_lead_time`、`evaluate_one_split` 可呼叫，main 有保護；前者是 L1 CV／OOF，後者會重新訓練且只回傳三政策 overall | 後者可作 L3 M1-NS lag=0 的委派基礎，但需補模型／OOF回傳、計時、追溯及族群輸出；不能用它跑 L2（會重訓），也不能直接用它跑 RQ3（會選成本門檻） |
| `experiments/06_fairness_groups.py` | 頂層 `dev_holdout()`、重訓 M1／M1-NS，只做教育／婚姻、0.5 比較 | 請負責人將分析抽為接收資料、分數、政策的純函式，再委派；補齊三政策及所有規定族群，不直接執行這支 CLI 當正式評估 |
| `experiments/07_group_threshold.py` | CLI 限已完成 L1-H run；頂層解析參數、重訓、核對 L1-H OOF；門檻搜尋有四位小數差距與高門檻同分規則，但結果尚未統一套最終名單上限 | 重用核准的搜尋規則，由負責人提供 L1／L3 OOF 輸入函式；正式結果透過 `Policy.group`／`apply_policy`，不能把 L1-H threshold 移作 L2 |
| `experiments/08_calibration.py` | L1 頂層實驗；已呼叫 `crossfit_calibrate`、`calibration_report`、`rank_change`，沒有保存供 L2 使用的全 OOF 校準器 | 抽出可接收分數的報表函式；另在 L1 全 OOF 擬合並保存最終 isotonic 校準器與雜湊，供凍結後只讀套用 |
| `experiments/05_pay_categorical.py` | 仍是只有說明、TODO 與 `_common` import 的骨架 | 尚無可採用的 M2-05 實作；不可宣稱已有模型或驗收通過 |
| `experiments/09_missed_analysis.py` | 有 `max_run`、`extra_features` 公式，但頂層執行；仍用 `feature_cols()` 預設含敏感屬性、全域 threshold、分群後各自五折、固定降幅 E、舊名單排序與 outputs/results | 這些與已凍結規則有差異，須由負責人修正並通過驗收、合併後才接入；不 import 現有模組來觸發頂層運算，不在 C9 偷改其分析邏輯 |
| `src/_protocol_runs.py` | 已有一般 L3 `evaluate_split`／`summarize_l3`，但只支援一般 prepare／單一 Pipeline，且摘要只讀 overall | 不能直接處理 05、D、E、F7 或無門檻 RQ3；需內部 adapter 及擴充報表，不改公開合約、不冒稱現有流程已支援 |
| `src/export.py`、`app/data.py` | export 明確只接受 L1-H，平台目前也限制 L1-H | C9 新增 L2 正式產生器，不放寬舊開發入口。平台接受 L2 是後續明確授權的整組切換工作，生成交接檔不等於目前頁面已可載入 |

`_common.py` 的相容模型入口已導向 C2 折內權重模型，但 `feature_cols()` 預設仍為 `sensitive=True`。LR0／LR1 在規格第 4.2 節只寫「原始／加衍生特徵」，**沒有寫敏感屬性設定**；指定的 03、06、07、08、09 中也沒有建置 LR0／LR1 的程式。因此不能把預設值當成已確認的 LR 實驗設定。第 8 節列出需組長核准的建議；沒有確認前阻擋這兩列的正式建置。

## 2. model_freeze 建置

### 2.1 程式位置、輸入與命令（擬議）

新增 `experiments/00_build_model_freeze.py` 為組長操作入口；純驗證、建置及序列化輔助碼可放 `src/model_freeze.py`，不得載入 L2。分兩階段，避免「草稿存在」被當成已核准凍結：

```text
python experiments/00_build_model_freeze.py prepare --rules-freeze-id g2a-20261116 --manifest <本機已確認的來源清單.json>
python experiments/00_build_model_freeze.py freeze --rules-freeze-id g2a-20261116 --model-freeze-id <組長指定ID> --manifest <已檢查的本機清單.json> --approved-at <組長核准時間>
```

- `prepare` 只使用 `dev_data()`：必要的全 24,000 人 fit、五折 OOF、折分配、政策與 L1 source run。可載入相符且有完整證據的現有紀錄；缺計時時不得填估計值，處理方式見 2.4／第 8 節。不產生可供 `final_test` 解鎖的 `model-*.json`。
- manifest 輸入每筆的 entry_id、角色、comparison_id、方法版本／實作 commit、model_record 或建置設定、source_run_id、政策來源與計時來源；含候選資格及組長選定平台模型的 L1 D7 證據。不從 `.env` 決定研究設計、不收任意測試集路徑。
- `freeze` 不訓練、不讀測試集；確認日期由組長提供、所有必要值已核准與工件驗證成功，才以不可覆寫方式建立 `outputs/frozen/model-<model_freeze_id>.json`。不以目前日期或是否填了字串取代人工核准。實作需拒絕既有 ID、缺值、重複 entry、無來源雜湊。
- 模型與政策的實際 ID 由 registry 產生／驗證；以下 entry 名稱只是穩定識別提案，不是假造 model_id。模型共用時沿用同一 model_hash，三政策各有自己的 policy_id、policy_hash、來源 run。

凍結頂層至少包含 `model_freeze_id`、`rules_freeze_id`、規則紀錄 SHA256、`approved_at`、核准者、建置 commit／套件版本／資料雜湊、`entries`。每筆保留現有 `frozen_entry` 需要的 `entry_id`、`role`、`comparison_id`、`source_run_id`、`model_id`、`model_hash`、`params_hash`、`columns`、`policy_id`、`policy_hash`；另記 `configuration`（演算法、完整參數、lag、sensitive、engineered、方法版本）、`protocol_policy`、features_hash、model_record 路徑、訓練 ID／OOF／折分配證據、訓練時間來源。有校準器另列路徑、calibrator_hash、擬合來源，不把它冒作基礎模型雜湊。

建置器以 `Registry.load_model`／`load_run` 驗證實際工件；寫入後逐筆以 `Registry.frozen_entry` 及新增的專用驗證器回讀。凍結內容不包含尚未產生的正式 predictions_sha256，也不允許占位雜湊。無門檻 RQ3 與獨立校準器目前不能完整通過既有 run／entry schema，這是後續實作缺口，需按第 8 節核准後增補驗證，不能虛構成本政策繞過。

### 2.2 政策記號與逐筆清單

- S：共同數值門檻 `Policy.single(0.5, None)`。
- N：固定名單人數 `Policy.top_n(0.2)`。
- C：模型自己的 L1 OOF 以 `choose_threshold(y, score, bill_at(df, 0), rules.scenario)` 選出的正式政策；主要情境為名單上限 0.2、損失比例 0.5、聯絡成本 50、挽回比例 0.2，實際值由規則讀取。
- 同一列列出三個 entry_id，就代表 **三筆模型＋政策紀錄**，不是一筆包三政策。每筆有各自 source_run_id 與 policy 證據；可共用同一模型、OOF及計時來源。
- RQ3 是第 4.5.1 節的明文例外：只評估 AUC／涵蓋率，不強加 S／C。以下 comparison_id 均為提案，正式寫入前由組長確認。

| entry_id（逐政策展開） | 角色／用途 | comparison_id | 模型與特徵設定 | 政策 |
|---|---|---|---|---|
| rq1-m1-s、rq1-m1-n、rq1-m1-c | 研究比較 RQ1 | rq1-sensitive | M1；XGB_TUNED；prepare(lag=0, sensitive=True, engineered=False)，23 欄 | S、N、C |
| rq1-ns-s、rq1-ns-n、rq1-ns-c | 研究比較 RQ1 | rq1-sensitive | M1-NS；同上 sensitive=False，19 欄 | S、N、C |
| rq2-lr0-s、rq2-lr0-n、rq2-lr0-c | 研究比較 RQ2／baseline | rq2-baselines | LR0；原始特徵；敏感設定待第 8 節確認 | S、N、C |
| rq2-lr1-s、rq2-lr1-n、rq2-lr1-c | 研究比較 RQ2 | rq2-baselines | LR1；與 LR0 相同原始組加 14 個衍生特徵，lag=0、engineered=True；敏感設定待確認 | S、N、C |
| rq2-dt0-s、rq2-dt0-n、rq2-dt0-c | 研究比較／報告 baseline | rq2-baselines | DT0；prepare(lag=0, sensitive=False, engineered=False)；決策樹 max_depth=4、class_weight=balanced、random_state=42，不調參 | S、N、C |
| rq2-ns-s、rq2-ns-n、rq2-ns-c | 研究比較 RQ2／baseline | rq2-baselines | M1-NS；原始 19 欄 | S、N、C |
| rq3-lag0 | 研究比較 RQ3（提早 1 個月） | rq3-lead-time | M1-NS；lag=0、sensitive=False、engineered=False，19 欄；共用既有 lag=0 模型 | 無門檻；AUC、5%～50% coverage_curve |
| rq3-lag1 | 研究比較 RQ3（提早 2 個月） | rq3-lead-time | M1-NS；lag=1，其餘同上，16 欄，只用序號 2～6 月資料 | 同上 |
| rq3-lag2 | 研究比較 RQ3（提早 3 個月） | rq3-lead-time | M1-NS；lag=2，13 欄，只用序號 3～6 | 同上 |
| rq3-lag3 | 研究比較 RQ3（提早 4 個月） | rq3-lead-time | M1-NS；lag=3，10 欄，只用序號 4～6 | 同上 |
| rq3-lag4 | 研究比較 RQ3（提早 5 個月） | rq3-lead-time | M1-NS；lag=4，7 欄，只用序號 5～6 | 同上 |
| f3b-sex | 研究比較 F3b | rq1-group-threshold | M1-NS，同一 model_hash | Policy.group：性別 |
| f3b-age | 研究比較 F3b | rq1-group-threshold | 同上 | Policy.group：年齡層 |
| f3b-sex-age | 研究比較 F3b | rq1-group-threshold | 同上 | Policy.group：性別×年齡層 |
| f3b-single-c | 研究比較 F3b 的單一政策對照 | rq1-group-threshold | 同上 | C（共用既有政策證據，不重新選） |
| f6-a-s、f6-a-n、f6-a-c | 研究比較 F6-A | rq2-f6 | A＝M1-NS，原始 19 欄 | S、N、C |
| f6-05-s、f6-05-n、f6-05-c | 研究比較 F6／候選，僅合格合併後列入 | rq2-f6 | M2-05；19 原始輸入欄，PAY 固定粗分類＋折內 One-Hot | S、N、C |
| f6-b-s、f6-b-n、f6-b-c | 研究比較 F6-B／候選，僅合格合併後列入 | rq2-f6 | M2-B；原始 NS 19 欄＋4 個代碼特徵，共 23 欄 | S、N、C |
| f6-c-s、f6-c-n、f6-c-c | 研究比較 F6-C／候選，僅合格合併後列入 | rq2-f6 | M2-C；B＋3 個金額特徵，共 26 欄 | S、N、C |
| f6-d-s、f6-d-n、f6-d-c | 研究比較 F6-D／候選，僅合格合併後列入 | rq2-f6 | M2-D；C 特徵組，兩子模型與分流規則一併保存 | S、N、C |
| f6-e | 研究比較 F6-E | rq2-f6 | M1-NS 分數，**不是新模型**，與 A 共用 model_hash | E 專屬 Policy.group，整體上限 20%，不是一般 F3b Recall 搜尋 |
| f7-iso-s、f7-iso-n、f7-iso-c | 研究比較 F7 | f7-calibration | M1-NS＋全 24,000 人 OOF 擬合的 isotonic 校準器；基礎模型同 hash、另記 calibrator_hash | S、N、校準分數自己的 C（來源處理待第 8 節確認） |
| f7-raw-s、f7-raw-n、f7-raw-c | 研究比較 F7 未校準對照 | f7-calibration | M1-NS 原始分數，沿用同一模型 | S、N、C |
| platform-s、platform-n、platform-c | 平台模型的兩個研究對照與正式用途（只有 platform-c 角色為平台） | platform-release | 組長按 D7 選定的 M1-NS 或合格 M2，具體選擇尚未發生；共用相應 model_hash | S、N、C；正式交接檔只用 C |

每個 comparison_id 內用相同測試客戶對齊，比較政策相同的列；不可混比 S 與 N，也不可重複計入跨 comparison_id 共用模型。可用外部索引連結同一來源模型的多個比較列，不能為了不同用途重訓不同模型卻填同一 hash。未合併或未驗收的候選以缺席原因列入建置報告，不產生假的 entries。

### 2.3 LR、DT 與候選的實作來源

LR0／LR1 採 C2 `make_model('邏輯迴歸', params=...)`，目前預設 max_iter=3000、class_weight=balanced，其他參數完整記錄於 configuration／params_hash；原始與衍生欄分別由 `prepare(..., engineered=False/True)` 取得。lag=0 的 14 衍生欄為 UTIL_1～6、PAYRATIO_1～5、N_DELAY、MAX_DELAY、UTIL_TREND。敏感欄四項為 SEX、AGE、EDUCATION、MARRIAGE；若採不含版本為 19／33 欄，若沿用含敏感版本為 23／37 欄，不能在未核准時自選。DT0 依本次明確要求用 NS 原始 19 欄，由新介面的決策樹建立，不用 L0 舊 builder。DT0、LR0、LR1 都不是 G2b 替換候選。

05 與 09 必須使用負責人**已合併且核對符合 G2a**的函式／模型 factory，記錄該實作 commit 與特徵清單，不能抄一份公式自行改。必要前處理放在可序列化 Pipeline 中；只做逐客戶確定性的特徵轉換可由 adapter 呼叫同一函式。輸出訓練與預測的欄位／順序必須一致。

候選的 XGBoost 完整參數沿用核准的基準設定並寫入清單；不因 C9 建置而重新調參。若合併實作帶有未核准的新參數或新特徵，先回報並阻擋該方法，不自動接受實驗檔常數。

- 05：PAY -2、-1、0、1、2 各一類，3～9 合一類；固定對照表、禁止目標編碼。現 main 尚無此實作。
- 09 B：從 `prepare(df, lag=0, sensitive=False, engineered=False)` 加 N_CODE0、N_CODE_M1、N_CODE_M2、RUN_CODE0；C 再加 PAYAMT_CHANGE、N_UNDERPAY、PAYAMT_CV。公式以凍結規則與負責人驗收後版本為準，不使用 09 現有預設含敏感欄。
- D：C 特徵；PAY_1～6 任一 >0 分有延遲。與其他方法相同五折，每折先切共同訓練／驗證，再在訓練折內分兩群 fit，合回同一驗證折 OOF。全 24,000 人 fit 的兩子模型可用既有 `GroupedModel` 保存；重載驗證兩群路由。分數直接合併，不另校準；報告跨群尺度未校準與前 20% 可能偏群的限制。
- E：M1-NS 的 C 門檻為 t0；有延遲固定 t0，無延遲在 0.05～0.95、步長 0.01 且 <=t0 搜尋。每候選 `Policy.group`＋20% 整體上限，以 D2 最終名單總成本最小、同成本較高門檻決定。保留截取前後結果；不參加 D7，也不直接上平台。
- D7 只排序 05、B、C、D：相同 L1 客戶／折，N 的 Recall 提升 >=0.01、AUC 不下降、性別與年齡 EO 增量各 <=0.02；多候選依凍結的五層排序（Recall增益、最大EO增量、合計EO增量、展開前欄數、05/B/C/D順序），全部用未四捨五入值。無合格者保留 M1-NS；正式決定由組長確認。

逾期依第 9 節：11/20 第一版不能跑時，由組長決定 Codex 與負責人合作；11/22 前未合併的 05／09 方法不參加 11/23 G2b 模型選擇，分析仍須完成並註明「未納入模型選擇」。不得把舊草稿視作合格方法，或在 G2b 後看到 L2 結果再補進候選。逾期方法的後續研究評估如何補獨立凍結清單，見第 8 節，原清單不可修改。

### 2.4 每個 24,000 人模型的計時與證據

擬以單調時鐘量 `fit_seconds`：只包住完整訓練資料的 Pipeline.fit（含需學習的前處理），不含 OOF、調參、SHAP、預測、磁碟寫檔。另記 `oof_seconds`、`policy_seconds`、`calibrator_fit_seconds`、`artifact_seconds`、總 wall_seconds、硬體／執行緒／作業系統／套件版本與開始結束時間，避免把不同成本混為訓練時間。

每個唯一模型、每個 lag 都有 training_count=24000 與真實 fit_seconds；D 計兩子模型訓練的總牆鐘時間，並附各子模型時間；E 重用 A 的訓練時間來源，不能記成新模型訓練 0 秒。F7 分開列基礎模型與校準器時間。各政策引用 training_timing_id／source_run_id，不因三政策重複 fit 或計算三次時間。

現有工件若沒有精確全量 fit 時間，標示未記錄，**不以 L4、CV 或檔案修改時間推算**；建議在 G2b 前重新以相同設定產生有計時的新 L1 工件及 OOF一致性證據，再以新實際 model_hash 凍結。是否重用既有工件與表格訓練時間的範圍，需第 8 節核准。L2 不為補計時重訓。

## 3. L2：final_eval 流程

擬新增 `src/final_eval.py`，正式入口維持規格原命令：

```text
python -m src.final_eval <model_freeze_id>
```

1. **先驗證、後取測試資料**：讀 approved model_freeze、load_rules 並核對 Markdown／JSON雜湊、rules引用、所有 entry 來源及資格。逐筆以 registry 核對 model.joblib 的 model_hash、params_hash、columns、資料雜湊、24,000 訓練 ID、source_run完成狀態、policy_hash／OOF／折證據；F7 另核 calibrator_hash，D 核子模型封裝。全部通過後才呼叫 `final_test(model_freeze_id)`；不得以缺失的 entry、備用 0.5 或較舊模型繼續。
2. L2 固定 6,000 ID，核對 seed42 分割、與訓練 IDs 不相交、各模型使用完全相同測試 IDs。特徵轉換只從凍結設定與已驗收 adapter 建立；不重新 fit、選特徵、選門檻、校準或調參。每個唯一模型（含校準器組合）預測一次並在同一執行內重用，按 ID 對齊，不按目前列順序拼接。
3. 每筆 `start_run('L2', ..., model_freeze_id=..., entry=...)`，以凍結模型與政策評估。S／N／C 分開 run；分數用完整精度，S 不限額、N 固定20%、C按正式上限。`evaluate` 產生 overall／groups／gaps；`apply_policy` 同時保留 pred_raw 與最終名單，用同一政策評估截取前／後並標註上限觸發。AUC由 score 計算，不能隨政策改變。
4. 族群包含性別、年齡層、性別×年齡層、教育、婚姻；分組沿用既有清理與 age_group，不偷偷新增切點。F3b 每次只用一種互斥完整 partition；缺群／無訓練違約者不能偷補門檻。族群列總人數／違約／未違約、Recall／FNR／FPR、Wilson 95%區間、D8少於93人標示；分母0空值並註明。`gap_ci`依凍結2000次、seed42計差距區間，需要模型差異時按相同 comparison_id 成對重抽。
5. RQ3 各 lag 只跑 AUC 與 `coverage_curve`（rates 0.05～0.50、間隔0.05），不調成本政策。預測只用當時可得月份，帳單涵蓋率事後用 `bill_at(test,0)`；不能用9月帳單選提前門檻。F7只套凍結校準器，另列 Brier、10箱校準曲線、排名／前N重疊，與 raw 比較，不換平台風險分數。
6. 產生第 4 節報表及平台交接檔；所有結果成功驗證後各自 `run.finish()`。另有一份執行 manifest 列全部 entry→run_id、來源 hashes、各狀態；只有所有必要項 complete 才可宣告整批正式執行完成。跨 entry 的相同政策 hash仍須依模型／comparison_id分辨。

## 4. 報告與交接輸出契約（擬議）

### 4.1 表 4-1、表 4-3 與混淆矩陣

**表 4-1 來源列**：每一 L2 模型／政策輸出 AUC、Accuracy、Precision、Recall、F1、fit_seconds；Precision／Recall／F1 的正類為違約(1)，分類名單使用最終政策結果。`evaluate` 沒有 Accuracy，後續新增的報表輔助函式從同一份 y 與 pred 的混淆計數計算 `(TN+TP)/n`，不能改動既有 evaluate 的固定回傳欄位或另寫名單演算法。

L2只有一個已凍結模型在seed42的測試結果，**沒有可估計的重複實驗標準差**。表 4-1 的「平均值 ± 標準差」建議另由 L3 的10個seed彙總提供，與 L2單次值分欄或分panel，不把五折CV／L4／10個seed混作L2誤差棒，不把同模型三政策當三次實驗。正式報表版型與主政策待第 8 節確認；保留所有政策原始列，避免先選好看的政策。

**表 4-3**：對每個非RQ3的 L2 模型／政策提供 label=0（未違約）、label=1（違約）兩列，包含 Precision、Recall、F1、support（真實該類人數），另提供 macro列。macro三指標為兩類指標的等權平均，support為總樣本數並明確標註，不當作平均每類人數。以相同混淆計數產生，記錄 labels=[0,1]；無預測正例等零分母處理待第 8 節確認，不默默把空值當0。另保存TN、FP、FN、TP及政策上限資訊，讓表與名單可核對。

**混淆矩陣**：至少產生平台選定模型 C 政策的 `confusion_matrix.csv` 及 `confusion_matrix.png`，列是真實類別、欄是預測類別，順序均為[未違約(0), 違約(1)]，圖內標原始人數，標題／圖註含 L2、model_id、policy_id、run_id、model_freeze_id。可另附比例圖，但不能替代人數圖；表4-3與圖必須由同一份最終名單計數生成。

### 4.2 位置與檔名

| 位置 | 檔名與用途 | 是否提交 |
|---|---|---|
| `outputs/frozen/` | `model-<id>.json`，核准後完整清單，不含預測 | 後續凍結PR只提交此核准紀錄；本卡不建立 |
| `outputs/models/<model_id>-<hash前12碼>/` | model.joblib、model.json、oof.csv、folds.csv；F7校準器及其來源以雜湊綁定 | 不提交 |
| `outputs/runs/<run_id>/` | run.json、threshold.json（有政策）、本機逐客戶分數／名單；只有平台正式 run 需要完整 predictions.csv | 不提交 |
| `results/<L1_source_run_id>/` | training_timing.csv、G2b候選彙總／缺席原因等不含逐客戶資料的證據 | 後續結果PR提交 |
| `results/<L2_entry_run_id>/` | overall.csv、groups.csv、gaps.csv、gap_ci.csv、raw_policy_overall.csv、raw_policy_groups.csv、raw_policy_gaps.csv、classification_report.csv、confusion_matrix.csv、table_4_1_row.csv、training_timing.csv、run.json副本 | 後續結果PR提交；無適用內容不造空假結果 |
| 同上RQ3 run | rq3_lead_time.csv、rq3_coverage.csv、run.json副本；不出成本／分類政策表 | 同上 |
| 同上F7 run | calibration_report.csv、calibration_bins.csv、calibration_curve.png、rank_change.csv，另含標準政策的報表 | 同上 |
| 同上平台C run | 另有 confusion_matrix.png；平台逐客戶檔仍只放 outputs/runs | 同上 |
| `results/<L3_entry_seed_run_id>/` | 同政策評估表、classification_report.csv、training_timing.csv、run.json副本，另記seed；不含逐客戶預測 | 同上 |
| `results/<summary_id>/` | evaluation_manifest.json、table_4_1_l2.csv、table_4_3_l2.csv、stability_by_seed.csv、stability_summary.csv、table_4_1_l3.csv、rq3_summary.csv、source_runs.csv | 擬議彙總目錄，與模型run區分；待第8節確認識別方式 |

所有 CSV 維持未四捨五入數值；報告／畫面才格式化。每列帶 level、comparison_id、entry_id、model_id、policy_id（適用時）、run_id、model_freeze_id、門檻來源。彙總表逐列或source_runs.csv引用全部source_run_ids，不捏造一個model_hash／policy_id代表多模型。不能把 `.env`、data、models、OOF、逐客戶預測或 sealed 檔案提交。

### 4.3 平台正式交接檔

只有 platform-c 的 L2 run 寫 `outputs/runs/<run_id>/predictions.csv`。欄位與既有交接契約一致：customer_id、risk_score、risk_rank、in_list、sex、age、age_group、education、marriage、limit_bal、bill_amt_latest、actual_default、data_split、model_id、run_id、policy_id、factor_1～3。data_split='test'，6,000個不重複 ID；風險分數完整精度保存，依分數降冪／同分ID升冪給rank，in_list只由同run的凍結正式政策計算。

使用相應特徵的已核對模型產生SHAP；One-Hot因素依規格帶符號合併。若平台選中05或D，必須先驗證其轉換／分流與SHAP路由支援，不能默用M1-NS模型因素。缺少可解釋支援時停止交接，不在看測試結果後改選模型。文字用「風險因素／模型判斷依據，不代表違約原因」。

run.json包含必要的git commit／git_dirty、套件、資料hash、level=L2、model_freeze_id／entry／凍結hash、實際model_hash／params_hash、policy_id／policy_hash／scenario、predictions_sha256、predictions_verified及verified_customers=6000。以重載模型對全部6,000人重算分數，核對完整精度容差（既有1e-9）、全部排名／名單與ID，再finish。threshold.json與兩個檔案識別一致，失敗不回退0.5。

平台換正式版本時只由組長在本機換 PLATFORM_RUN_ID；沿用run_id與predictions_sha256快取鍵，整組切換，不混用L1-H模型或門檻。現app/data.py的L1-H限制需後續獨立核准修改，不能在本設計PR中更動。

## 5. L3：10個seed的穩定性與彙總

擬新增 `src/stability_eval.py`，命令提案 `python -m src.stability_eval <model_freeze_id>`，不得依L2表現挑子集或重新搜尋參數。只有凍結後才呼叫 `outer_splits(model_freeze_id, seeds=range(42,52))`，由此取得seed、train、test，不自行重寫切分。

每seed按凍結的唯一模型／特徵設定重建，參數固定、類別權重每次fit重算，重新產生該seed的24,000人五折OOF及全量模型；三種政策共享該seed模型，不為每政策重新fit。模型演算法random_state依凍結configuration，不把切分seed無聲改作模型參數；OOF折分seed與切分seed記錄清楚。每個seed／entry各自新run_id，model_id／model_hash與政策來源皆為這次實際工件，不能套凍結L2模型hash，即使seed=42也照L3規則記錄。

| 政策／方法 | 每seed訓練端 | 測試端 |
|---|---|---|
| S | 固定0.5，無上限，不重新選 | 用該seed分數直接apply_policy |
| N | 固定20%，不重新選 | 依當批完整精度排名取floor(0.2×6000) |
| C | 該seed OOF＋bill_at(train,0)重新choose_threshold，保存新policy與OOF證據 | 只套該新政策；不依test成本修正 |
| F3b | 以該seed M1-NS OOF的C原始名單Recall作目標；每群網格差距先round4，同分高門檻；新policy_id | 一次一種partition，20%上限，原始／最終均評估 |
| F6-E | 依凍結E選擇程序重算該seed t0及無延遲群門檻，不能套一般F3b搜尋 | 不改有延遲群t0、整體20%上限；不作模型替換 |
| RQ3 | 每lag依固定參數重新fit；不選成本政策 | AUC與coverage_curve；事後帳單涵蓋率用9月帳單 |
| F7 | 每seed僅用train的OOF作交叉擬合評估及擬合供test使用的完整校準器；另存實際calibrator_hash | 只轉換test分數，校準後政策依其種類處理；不替換平台 |

委派方式：03的evaluate_one_split作一般lag=0邏輯基礎，需先把回傳模型、OOF、計時與政策持久化補齊，避免另重訓一份模型才記錄；RQ3重用03的月份／rates設定與coverage流程但不呼叫帶成本政策的evaluate_one_split。06接收各seed分數與三政策；07接收該seed的OOF決定族群門檻；08接收該seed的OOF／test分數；05／09由已合併adapter提供特徵與模型factory。頂層有副作用的檔案必須先由負責人抽成函式並加main保護，不能用subprocess執行目前CLI來假裝L3已接好。

每seed用2.4同樣計時範圍記fit_seconds、OOF／政策／校準器時間；彙總按comparison_id、模型配置、政策種類、lag、規則版本分組，不按每seed新policy_id切成十個單筆組，也不把不同政策混在一起。輸出10筆逐seed值及AUC、Accuracy、Precision、Recall、F1、fit_seconds的mean、std、min、max、n_valid；建議std採樣本標準差ddof=1（與現summarize_l3一致），不是CV函式的母體標準差。缺seed不發布完整「10次」結果，補跑失敗seed後再做新彙總。

表4-1格式欄由mean±std生成，原始值另存。RQ3只彙總AUC／涵蓋率及計時，不為湊欄位新增成本門檻。L3各次測試資料互有重疊，也與L1選擇資料重疊，是穩定性／切分敏感度分析，不能寫成十次獨立驗證或宣稱外部驗證。

## 6. 執行順序、時間與組長電腦

| 階段 | 時間／位置與執行者 | 前置條件與輸出 |
|---|---|---|
| 設計審核 | 本卡，只讀程式、撰寫文件 | 組長先確認第8節；本次不啟動下列工作 |
| 後續實作與無資料測試 | 核准後另卡／PR，Codex；假資料與Mock | 驗證凍結前取test必敗、hash錯誤、失敗不覆寫、計時與報表一致性；組員確認各adapter |
| L1建置／G2b準備 | 組長電腦，11/22候選合併截止前後整理；由組長授權Codex執行 | 只用L1，全部必要工件、計時、候選資格及清單草稿；不預先讀L2 |
| G2b核准／寫入 | 原排11/23，組長電腦，由組長確認模型與核准時間 | freeze驗證後一次寫model-<id>.json；提交核准JSON前不啟動正式測試 |
| L2 | 原排11/24～25，組長電腦授權執行final_eval一次 | 僅讀凍結模型，一批6,000人；先完成來源驗證，再評估／交接 |
| L3 | 同期，組長電腦授權執行10個seed | 固定設計；完成逐seed run及彙總，不依結果改任何設定 |
| L4開封 | 原排11/25～26，G2b存在且引用相同規則後，組長明確授權 | 只呼叫open_sealed(rules_freeze_id,model_freeze_id)，核對封存hash、30,000 ID各一次、折對應，建立新的L4 run；不直接讀sealed分數 |
| 平台整組切換 | 正式L2交接完成並驗證，組長本機 | 後續核准平台接受L2後換run_id、錄展示；不公開部署 |

時間只作排程估算，不是假造本次量測。凍結的V7基準為單一M1-NS全24,000人fit約9.052秒、OOF約119.769秒（WSL2／32 logical CPUs）；本次不再試跑。若所有候選合格，約有13個唯一基礎模型：M1、M1-NS、LR0、LR1、DT0、另4個lag、05／B／C／D；D含兩個子模型，F7另計校準器，E不新增模型。

L1預算用「各模型實測fit＋OOF＋政策＋驗證時間」相加。僅用13×(9.052＋119.769)得到約28分鐘作粗略XGB等效運算量，**不代表LR、05、D同速**。L3同式乘10約4.65小時，另加校準、族群bootstrap、SHAP／I/O與失敗重試；建議預留一個工作日，依L1計時更新排程但不依效能指標改設計。L2沒有fit，主要為各模型預測、族群區間與平台SHAP，暫預留1～2小時；此為待核准排程額度，沒有本次執行證據。L4已完成的執行事件可引用既有執行紀錄，不重跑L4；3.77小時是原L4估計，不能拿來當C9全套耗時或表4-1訓練時間。

## 7. 安全、驗證與失敗處理

- 新CLI與可import模組不得有頂層讀資料／訓練副作用。建置器只透過dev_data取L1；L2／L3入口在任何test資料載入前驗證整份已核准freeze；測試以Mock確認freeze不存在、不核准、任一模型／政策／校準器hash不符時不觸及資料loader。原始資料檔包含30,000人，dev_data內部為切分而讀檔不等於授權取用test；禁止應用程式繞過protocol自行取test列。
- 既有 `python -m src.protocol --check-g2b` 與整合測試需通過；新增入口必須納入檢查。這是程序防呆，不是對任意讀檔的沙箱或加密保證。L0只歷史重現；封存L4是既有例外，不能藉此讓一般分析讀sealed。
- 不讀舊outputs/predictions.csv、不以L1-H挑模型。正式模型只能來自全24,000人L1與核准OOF，政策不得用另一run的預設門檻。測試資料只在G2b後取一次做固定設計評估，不能根據錯誤率或成本再改政策。
- 後續應以人工合成資料驗證：所有模型／政策識別與hash、F7校準器證據、D子模型路由、RQ3無未來欄位、完整名單排序／同分ID、6,000人全數重載核對、兩類報表／混淆矩陣與overall一致、L3每seed證據及計時、輸出不含逐客戶敏感資料。這些是未來驗收，不宣稱本卡測過尚不存在的程式。
- 依5.3.1(4)：執行中錯誤即停止該批，當前run標failed並保留原因／時間／可查核狀態；不能finish、不能當正式結果、不能覆寫已完成run或凍結紀錄。未開始entry記未執行，先完成的run也不讓整批冒充成功。
- 重試建新run_id，manifest保留前次失敗與新來源連結；已完成的正式結果不偷偷替換。建議重試只重算失敗／未執行entry，是否沿用已完成entry須由組長明確核准，見第8節。若是實作錯誤需更動程式／研究設定，由組長確認影響、記錄原因與差異；需要新freeze就用新ID，不能編輯舊JSON再沿用舊ID。
- L4開封失敗不能修改seal.json或封存檔；依既有規則新建開封run，保留原因，同份封存不能有兩個未說明的完成正式開封run。開封結果只報告，不回饋模型、門檻、特徵或任何研究選擇。

## 8. 待組長決定（核准前不得以預設值執行）

| 編號 | 未明確／需取捨事項 | 建議與理由 |
|---|---|---|
| Q1 | LR0／LR1敏感屬性，規格未寫、現main沒有這兩支實驗建置程式 | 建議兩者均NS（19／33欄），與新增DT0及M1-NS的baseline比較一致；若要保留期中含敏感RQ2版本，需另列23／37欄研究對照，不能稱已確認期中設定。請組長指定 |
| Q2 | 表4-1要求「測試集效能平均±標準差」，但L2規格只執行一次；主表採哪政策／模型 | 建議分L2單次panel與L3十seed mean±sample-std panel，主列用C並附S/N完整表。請確認報告標題與DT0/LR/M1-NS等主列；不造L2標準差、不對同一test重訓湊重複 |
| Q3 | RQ3無政策與「每模型三標準政策」及現registry每entry要求policy證據的矛盾 | 建議採4.5.1 RQ3特例、無門檻entry與run，不填假policy；新增明確評估種類及專用驗證。lag0共用M1-NS模型，不另訓練一個假版本 |
| Q4 | F7的C門檻來源：L1可評估分數是交叉擬合分數，但L2用全OOF擬合校準器；L3未明寫F7校準器重擬 | 建議C只由交叉擬合校準OOF依D2選，保存各折來源；全OOF校準器只用於test，另外hash；L3每seed重做同流程，禁止以全OOF擬合後同資料轉換的分數選政策。請確認與現schema擴充方式 |
| Q5 | 11/22未合併05／09仍須完成研究，如何納入後續L2/L3而不污染G2b | 建議不入平台候選／原release清單；後續只在新核准研究清單評估，明列未納入模型選擇及看到正式結果後補作的限制。若方法已合併但不合凍結規則，同樣阻擋，不能視為「已合併就合格」 |
| Q6 | 訓練時間範圍與既有model沒有fit計時 | 建議主表用全24,000人Pipeline.fit秒數，OOF／調參另列；G2b前缺時間者重建新L1模型紀錄，不在L2補fit。F7主表分列基礎fit與calibrator fit，D列兩子模型總時間，E引用基準來源 |
| Q7 | 每類Precision等零分母與macro算法的缺值處理 | 建議保存空值＋未定義旗標，macro任一類未定義則標無法完整計算，而不是默默忽略該類；如範本要求sklearn zero_division=0，請明確核准並在表註揭露。族群零分母空值規則維持不變 |
| Q8 | 多entry部分成功後重試範圍、彙總識別及同模型不同政策來源run的重用 | 建議執行manifest串接已完成run，失敗entry新ID補跑；彙總用summary_id＋完整source_run_ids，不能假裝單一model run。具體schema與CLI可在後續實作審查確認，不能覆寫complete |
| Q9 | 候選D／05若勝出平台，現有SHAP轉換與正式L2平台載入支援 | 建議G2b前把候選的重載／特徵／SHAP可用性列工程驗收，組長確認修復或時程處理；不得看L2後因介面困難改平台模型。平台接受L2另做範圍明確的切換，不在本卡放寬L1-H限制 |
| Q10 | comparison_id／entry名稱、模型凍結ID、核准者與時間、排程預留 | 建議採本文用途分組及共用模型來源，凍結ID／批准時間由組長提供；時間先用L1實測更新。本文未指定實際平台winner、沒有建立任何model_freeze |
| Q11 | L3 seed42重新fit可能與L1模型產生相同實際檔案hash，但現ProtocolRun.update會拒絕L3與凍結模型hash相同 | 建議區分「有獨立fit與新model_id／來源證據，但位元相同」與「直接借用L2工件」；由組長確認如何調整驗證契約，保留真實hash，不能改模型參數或序列化內容去湊不同hash。此點不代表已允許跳過L3重訓 |

Q1、Q2、Q3、Q4、Q6、Q7等會影響可執行契約；核准後應先在後續PR將必要輸出契約與例外同步到整合規格，不修改既有G2a凍結內容。如確需改研究規則，遵守新rules_freeze_id及受影響實驗重跑規則。本文件不是擅自修訂已凍結研究設計的授權。
