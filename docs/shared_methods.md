# G2a 凍結方法共用函式（C9a）

實作位置：`src/frozen_methods.py`。**組員程式只負責 L1 比較分析、圖表與報告解讀，不要自己重寫這些公式、分群或門檻搜尋。** L1 與後續 L2／L3 使用同一份共用實作，正式執行不呼叫組員實驗程式。

所有函式的 `rules_freeze_id` 都是僅限關鍵字的參數，預設 `g2a-20261116`。每次使用均透過 `src.protocol.load_rules()` 驗證並讀取凍結紀錄；不讀 `.env`、不讀原始資料、不自動訓練或執行實驗、不在 import 時讀規則。模型參數、PAY固定對照、候選特徵清單、搜尋網格、名單上限、主要成本情境及Recall差距四捨五入位數都從凍結JSON讀取，不接受臨時覆寫。式中的代碼0／-1／-2、加1及ddof=1是已核准公式本身，不是新增的可調研究設定。若 `feature_formulas` 不符支援公式，直接報錯，不以eval執行字串。

## 共同使用方式

從專案根目錄執行組員的 L1 分析程式，依需求引用下列函式；本文件範例不是已執行的結果。

```python
from src.protocol import dev_data, load_rules, make_model, oof_predict, bill_at, apply_policy
from src.frozen_methods import (
    prepare_pay_categorical, make_pay_categorical_model,
    prepare_missed_features, delay_partition, make_grouped_model,
    search_no_delay_threshold, search_group_thresholds,
)

rules = load_rules('g2a-20261116')
df = dev_data()  # 只能以 L1 的24,000人作選擇；不要自行讀測試集
# 下列所有OOF共用此折數及seed；不可先分群再分折。
cv = dict(folds=rules.cv['n_splits'], seed=rules.cv['random_state'])
```

資料以 UCI 客戶 ID 為索引。轉換函式回傳 `(X, cols)`，不修改輸入資料；X保留客戶順序與資料層級attrs，特徵沒有標籤、ID或敏感四欄。`oof_predict` 會保留折分配證據。轉換本身是逐客戶確定性計算，無需標籤；需要學習的前處理只在模型Pipeline的fit內執行。

## 05：PAY 類別編碼

- `prepare_pay_categorical(df, *, rules_freeze_id=...)`：df為原始欄位DataFrame。以 `prepare(..., lag=0, sensitive=False, engineered=False)` 的19欄為基礎，將PAY_1～6映射為pandas類別欄；回傳 `(X, cols)`。
- 對照固定且不依資料／標籤學習：-2、-1、0、1、2各自一類，3～9合為「延遲三個月以上」。實作直接套凍結JSON的mapping，不另建立另一份可能漂移的數字表。出現未知代碼（含小數）、缺值或非有限數值就報錯，不默默歸類。代碼0與-2不解釋成確定的金融意義。
- `make_pay_categorical_model(*, rules_freeze_id=...)`：回傳尚未fit的Pipeline。PAY欄OneHotEncoder與其他數值欄StandardScaler均在每個訓練折fit；XGBoost參數讀dev_base.params，類別權重由每次fit的標籤計算。禁止目標編碼。
- 驗證折可能出現合法但該訓練折未見的映射類別，One-Hot以handle_unknown='ignore'處理；這不允許對照表外原始PAY代碼，原始碼驗證在prepare時先完成。

```python
X, cols = prepare_pay_categorical(df)
oof_05 = oof_predict(make_pay_categorical_model, X, df['default'], **cv)
model_05 = make_pay_categorical_model().fit(X, df['default'])
# 預測前也先prepare_pay_categorical；不要直接把原始數字PAY送入模型。
```

## 09 B／C：七個新特徵

`prepare_missed_features(df, *, method, rules_freeze_id=...)`：method必填，接受 `'B'`、`'C'`、`'D'`。B回傳23欄；C及D回傳相同26欄。基礎為M1-NS的19原始欄，欄位順序核對凍結候選清單，回傳 `(X, cols)`。

| 特徵 | 已凍結公式 |
|---|---|
| N_CODE0 | PAY_1～6等於0的月數 |
| N_CODE_M1 | PAY_1～6等於-1的月數 |
| N_CODE_M2 | PAY_1～6等於-2的月數 |
| RUN_CODE0 | 依PAY_1到PAY_6順序，等於0的最長連續月數 |
| PAYAMT_CHANGE | (PAY_AMT1 − mean(PAY_AMT2～6)) / (mean(PAY_AMT2～6) + 1) |
| N_UNDERPAY | m=1～5之PAY_AMTm < BILL_AMT(m+1)的次數，等於不計 |
| PAYAMT_CV | std(PAY_AMT1～6, ddof=1) / (mean(PAY_AMT1～6) + 1) |

B只加前四欄，C／D再加後三欄。需要標準化時由C2模型Pipeline在各訓練折fit，不在整份資料上先fit。若公式分母等特殊輸入造成非有限結果就報錯，不偷偷更改公式或補值。

```python
X, cols = prepare_missed_features(df, method='B')  # C改method='C'
factory = lambda: make_model('XGBoost', params=rules.dev_base['params'])
oof_B = oof_predict(factory, X, df['default'], **cv)
```

## 09 D：分群模型

- `delay_partition(df, *, rules_freeze_id=...)`：只需PAY_1～6，回傳與df索引相同的Series，name=`有無延遲`；任一PAY>0為`有延遲`，否則為`無延遲`。拒絕缺值／非有限PAY。只用作分流，不從標籤學習。
- `make_grouped_model(*, rules_freeze_id=...)`：回傳未fit的 `FrozenGroupedClassifier`。也可直接建立該類別；其建構參數僅有rules_freeze_id。輸入X必須由 `prepare_missed_features(..., method='D')` 取得，欄位順序需完全一致。
- `.fit(X, y)`：在傳入的訓練資料內分兩群，各fit一個讀凍結XGB_TUNED參數的子Pipeline；各群的scale_pos_weight只用各自訓練標籤計算。回傳自身；已fit模型可由`.models_`檢查兩個子模型。
- `.predict_proba(X)`：依每位客戶的PAY路由，回傳與X順序一致的n×2分數陣列，`.classes_=[0,1]`。`.predict(X)`回傳較高分數的類別，**不是正式預警名單**；名單仍需apply_policy。預測只含一群也可運作。
- 模型可clone與joblib保存／重載。OOF時先由oof_predict切共同五折，再在每折fit內分群；禁止先分兩群後各自五折。若某訓練折某群為空或缺任一類標籤會報錯，不退回另一模型，不自行改折或平滑權重。

```python
X, cols = prepare_missed_features(df, method='D')
oof_D = oof_predict(make_grouped_model, X, df['default'], **cv)
model_D = make_grouped_model().fit(X, df['default'])
partition = delay_partition(df)
```

存檔最短例（路徑僅示意，正式模型應經registry保存實際工件與OOF證據）：

```python
import joblib
joblib.dump(model_D, 'outputs/d-development.joblib')  # 本機檔，不提交
restored = joblib.load('outputs/d-development.joblib')
score = restored.predict_proba(X)[:, 1]
```

D跨群分數直接合併，不另校準；報告要列「跨群分數尺度未校準、前20%名單可能偏向某一群」限制。

## 09 E：無延遲群門檻搜尋

`search_no_delay_threshold(df, oof_score, *, rules_freeze_id=...)`：df含default、PAY及帳單欄；oof_score是**同一M1-NS模型／同次訓練**的OOF Series，索引與df完全一致。函式不訓練、不讀任何其他run或預設threshold檔，呼叫者必須提供正確OOF來源。

1. 由df標籤、oof_score、bill_at(df,0)及凍結主要Scenario呼叫choose_threshold取得t0。
2. 有延遲群固定t0；無延遲群只搜尋凍結no_delay_grid中<=t0的點。
3. 每點用Policy.group、凍結整體名單上限及apply_policy產生最終名單；成本由protocol.evaluate計算，不另寫公式。選最終總成本最低，同成本選較高無延遲門檻。

回傳dict：

| key | 內容 |
|---|---|
| policy | 選定的Policy.group，selection='group-cost'，含scenario與OOF內容雜湊等來源 |
| candidates | 每個候選一列：threshold（無延遲門檻）、t0、raw_cost、final_cost、raw_count、final_count、cap_triggered、policy_id、selected |
| t0／base_policy | M1-NS主要情境的基準門檻與單一正式政策 |
| partition | 有無延遲Series，後續apply_policy時必須一起傳入 |

```python
# oof_ns須事先用M1-NS原始19欄模型取得，不可用B/C/D的分數代替。
result_E = search_no_delay_threshold(df, oof_ns)
policy_E = result_E['policy']
final_list, info = apply_policy(oof_ns, policy_E, partition=result_E['partition'])
search_table = result_E['candidates']
```

raw_cost是不截上限的原始名單成本，final_cost才是選擇依據；全部候選表保留，不只回傳最好的那列。上限觸發後不同門檻可能有相同最終名單，照實報告。E不產生新模型分數、不參加G2b模型替換、不直接套用平台。

## F3b：族群門檻搜尋

`search_group_thresholds(y, oof_score, bill, partition, *, partition_name, rules_freeze_id=...)`：四個輸入都是同索引同順序Series；bill須為 `bill_at(df,0)` 的非負預警帳單。partition每人恰有一個非空標籤，一次只傳一種分組。partition_name只能用凍結的性別、年齡層、性別×年齡層；不接受DataFrame多組分法、缺值、索引不合或字串化後名稱衝突。

先由同模型同OOF的主要成本政策取得t0；目標Recall是 `score >= t0` 的**截取前**原始單一名單整體Recall。各群在凍結網格中找與目標Recall的絕對差最小者，差距依凍結位數（目前4）四捨五入，同差距選較高門檻。若整體或任一群沒有違約者就報錯，不能算Recall或偷補門檻。

回傳dict：policy（含凍結上限的Policy.group）、candidates（逐群逐網格的group、threshold、recall、target_recall、gap、rounded_gap、total_count、default_count、nondefault_count、selected），另有target_recall、t0、base_policy。三個count分別是該群總人數、違約與未違約人數。

```python
partition = df['SEX'].map({1: '男性', 2: '女性'})
result_F3b = search_group_thresholds(
    df['default'], oof_ns, bill_at(df, 0), partition, partition_name='性別')
final_list, info = apply_policy(oof_ns, result_F3b['policy'], partition=partition)
```

使用性別／年齡欄分群不等於把敏感欄加回模型特徵。截取上限前的Recall接近目標，不保證截取後仍成立；正式比較需同列原始／最終結果。這是參考Hardt的簡化研究做法，不是完整Hardt實作，也不是平台政策。

## 證據與使用界線

- 兩種搜尋都只接受訓練OOF作選擇。帶有level attrs時只接受L1／L3；拒絕L1-H、L2與L4。人工測試Series可無attrs，但「沒有attrs」不是正式使用授權，也不能證明分數是OOF或M1-NS。
- 傳入Series須為完整互斥分組及相同有序客戶IDs；分數必須有限且在[0,1]。E需同時有兩群。L3及L2的資料只允許後續已核准的正式入口在G2b後取得，本卡不實作final_eval／stability_eval。
- 回傳policy已有policy_id／policy_hash及OOF內容雜湊，但不是完整registry threshold.json，也不代表有model_id／run_id或模型來源驗證。組員仍須用既有run／registry紀錄模型、OOF、折分配與政策；不要虛構識別或覆寫完成紀錄。E的group-cost後續L3處理不能誤用一般F3b Recall搜尋，將在正式入口工作卡串接。
- 本卡不修改experiments/、既有公開protocol介面、規則凍結與封存，也不執行正式實驗。測試只用人工資料，包含真實XGBoost子模型的折內fit、路由及joblib重載，沒有對真實客戶產生預測。
