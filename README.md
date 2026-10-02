# 内部评分卡服务

后端服务：上传 CSV 开发样本 -> 后台自动建卡（等频切箱、相邻合并、WOE/IV、
牛顿法逻辑回归、PDO 分值换算），每一步中间结果随版本留存；之后用同一版本
对单条或批量申请人在线打分。仅 HTTP 接口，无前端。

* Python 3.12 + FastAPI + NumPy
* 分箱合并、WOE/IV、逻辑回归（牛顿迭代）、KS/AUC 全部自行实现，
  不依赖 scikit-learn / statsmodels
* PostgreSQL 16 存储作业与版本；默认也可用内存存储跑 pytest

## 目录结构

```
app/
  parsing.py    样本解析与作业前硬性校验
  binning.py    分箱引擎：等频初箱 + 相邻合并 + WOE/IV
  regression.py 牛顿迭代逻辑回归
  scoring.py    评分换算与在线落箱打分
  metrics.py    KS / ROC-AUC
  card.py       建卡管线编排与评分卡文档
  scheduler.py  后台作业调度（线程池，作业间数据隔离）
  storage.py    版本存储（内存 / PostgreSQL）
  api.py        FastAPI 接口层
tests/          pytest 性质测试 + 可手算小样本
```

## 启动

```bash
docker compose up --build
# API: http://localhost:8000  文档: http://localhost:8000/docs
```

本地（不启 Postgres，内存存储）：

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
```

## 建卡

`POST /jobs`（multipart/form-data）：

| 字段 | 说明 | 默认 |
| --- | --- | --- |
| `card_name` | 卡名（同名多次建卡产生新版本） | 必填 |
| `file` | CSV 开发样本，0/1 标签列 + 特征列 | 必填 |
| `target_col` | 标签列名；缺省自动识别 `target/label/y/bad/...` | 自动 |
| `features` | 逗号分隔的入模特征清单 | 缺省按 IV 阈值筛选 |
| `iv_threshold` | 自动筛选 IV 下限 | 0.02 |
| `min_bin_pct` | 每箱最小样本占比 | 0.05 |
| `base_score` / `base_odds` / `pdo` | 评分换算参数 | 600 / 50 / 50 |

作业开始前拒绝（HTTP 400，作业不入库）：样本 <500 行、标签不是 0/1 或
全同一类、PDO 非正、指定的入模特征不存在、最小箱占比非法。

```bash
curl -X POST http://localhost:8000/jobs \
  -F card_name=retail_a \
  -F min_bin_pct=0.05 \
  -F file=@dev_sample.csv
# {"job_id":"...","status":"queued"}

curl http://localhost:8000/jobs/$JOB_ID      # queued/running/succeeded/failed
curl http://localhost:8000/cards/retail_a   # 缺省最新版本，?version=N 指定
curl http://localhost:8000/cards/retail_a/versions
```

卡文档包含：每特征分箱边界、各箱好/坏计数、坏率、WOE、IV、合并轨迹
`merge_trace`、初始箱 `initial_bins`、回归系数、牛顿迭代次数、每箱分值表、
KS/AUC/特征 IV、截距分摊说明。

### 版本号分配与并发语义

* 版本号在建卡**完成、写库时**分配（按完成先后），同一卡名的分配经
  PostgreSQL 事务级咨询锁串行化；不同卡名锁键不同，互不阻塞。
  内存存储用进程内同一把锁实现同样语义，两种后端对外表现一致。
* 版本分配、写卡、作业状态更新是同一个数据库事务：作业成功则版本与
  状态同时生效；建卡途中失败则什么都不留——不留半截版本，失败作业
  不占号，成功作业的版本号恒为稠密的 1..N，作业记录上的版本号与
  该版本卡里的建卡参数一一对应。
* 代价：版本号顺序是"完成顺序"而非"提交顺序"。并发提交时，晚提交但
  先完成的作业会拿到更小的号；"最新版本"始终指最近完成建卡的版本。
* 另一种做法（提交时即分配版本号）版本顺序与提交一致、提交时即可预知
  版本号，但建卡失败的作业会留下永久空号（版本表缺号）；要避免空号就
  得先写占位版本行，又与"失败不留半截"的要求冲突。且并发下"提交先后"
  本身也要靠加锁来定义，实际收益有限，故不采用。

### 分箱规则

1. 数值特征先按唯一值贪心等频切成不超过 20 个初始箱（区间 `(lower, upper]`）；
   类别特征按坏样本率升序排列每个类别。
2. 相邻合并顺序固定：
   1) 合并好/坏计数为 0 的箱；
   2) 合并样本占比低于 `min_bin_pct` 的箱（合并目标用确定性平局规则，
      保证同一份数据每次结果一致）；
   3) 反复合并 WOE 单调性违例最明显的相邻箱对。
3. 数值 WOE 方向由箱代表值与 WOE 的相关系数决定；类别特征坏率升序 =>
   WOE 非升。每箱好、坏样本均非零。
4. 缺失值始终单独成一箱，不参与相邻合并，也不纳入数值 WOE 单调性判断；
   缺失箱若只有单一类别，WOE/IV 使用 +0.5 修正（原始计数仍保留，
   `counts_corrected_0p5=true`），避免无穷分值。

### 评分换算

```
odds = 好/坏 = (1-PD)/PD
factor = PDO / ln 2
offset = base_score - factor * ln(base_odds)
score  = offset - factor * logit(PD)
```

截距与 `offset` 按入模特征数平均分摊，每个特征各得一份基准分与截距份额；
每箱分值 = `per_offset - factor*(intercept_share + coef*WOE)`。
全部箱分之和即为总分；总分增加一个 PDO，odds 恰好翻倍。

## 在线打分

```bash
curl -X POST http://localhost:8000/score/retail_a \
  -H 'Content-Type: application/json' \
  -d '{"features":{"income":8200,"age":51,"city":"A","housing":"own"}}'
```

返回总分、PD、好/坏 odds 与每个特征的落箱编号/箱标签/WOE/得分。
`?version=N` 指定版本，缺省最新。响应中的 `card_version` 与本次算分
实际使用的卡在同一次读取中确定（单条、批量一致）：缺省最新时即使读取
间隙有新版本落库，报的版本号也不会与算分用的卡错位。

* 训练时没见过的类别：落专用箱 `UNSEEN_CATEGORY`（WOE 视为 0，得该特征
  基准分），返回中 `has_unseen=true` 且在 `unseen_features` 标出，
  不静默并入任何训练箱。
* 数值超出训练范围：落入端箱并标记 `out_of_range`。
* 训练时存在缺失的特征按缺失箱打分；训练从未缺失而打分缺失会报错该条。

批量：`POST /score/{name}/batch`，`{"applicants":[{...},{...}]}`，
单条错误只影响该条（`results[].ok=false` 带错误信息），其余正常返回。

## 测试与手算样本

```bash
pytest            # 内存存储，不需要数据库
```

PostgreSQL 集成测试（真实库，覆盖并发建卡版本号、边建卡边打分的版本
一致性、作业成功/失败的原子性）：

```bash
docker compose up -d db
DATABASE_URL=postgresql://scorecard:scorecard@localhost:5432/scorecard \
    pytest tests/test_postgres_storage.py
```

未设置 `DATABASE_URL` 时该文件整体跳过（pytest 摘要里可见原因）；设置了
但连不上库会直接报错失败，不会静默跳过。

覆盖性质：平均预测 PD == 实际违约率（1e-6 内）；标签整体取反后 WOE 变号、
IV 不变；样本复制一份后分箱/WOE/系数不变；箱分之和 == 总分；总分 +PDO
时 odds 翻倍；PD 随总分严格单调且恒在 (0,1)；数值特征 WOE 单调；
未见类别标注；作业前拒绝；并发作业隔离；版本与批量接口。

`tests/data/sample_development.csv` 是 24 行小样本（8 坏 16 好），
前两个特征 `city`、`housing` 的 WOE 可直接手算，测试
`test_city_woe_iv_hand_calc` / `test_housing_woe_iv_hand_calc` 已写入
期望值，例如 `housing`：

| 箱 | 好 | 坏 | WOE |
| --- | --- | --- | --- |
| own | 7 | 1 | ln(3.5) ≈ 1.2528 |
| mortgage | 5 | 3 | ln(5/6) ≈ -0.1823 |
| rent | 4 | 4 | -ln2 ≈ -0.6931 |

`age` 列另有 1 个缺失值，用于核对缺失单独成箱。
