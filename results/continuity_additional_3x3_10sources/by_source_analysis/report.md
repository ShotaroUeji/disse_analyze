# 音源ごとの QNX(2)・Sadj 解析

入力: `results/continuity_additional_3x3_10sources/summary.csv`

この解析は既存の `summary.csv` を読み込むだけで、モデル推論や音声生成は行っていない。各音源IDについて、9音響条件（3部屋×3 T30）の平均・標準偏差・最小値・最大値を求めた。

## 指標の読み方

- QNX(2): 高いほど、物理近傍が埋め込み近傍として保存される。
- Sadj: 低いほど、物理的に隣接する点の埋め込み距離が均一である。
- `qnx_t30_slope` と `sadj_t30_slope` は、部屋サイズを平均して T30=0, 0.3, 0.6 s に対して線形近似した記述的傾き。

## 音源イベント別の上位・下位

### DISSE / azimuth

- QNX(2) 上位: 104937 (0.683), 104403 (0.678), 103289 (0.659)
- QNX(2) 下位: 106039 (0.602), 10203 (0.620), 105490 (0.625)
- Sadj 上位（低い順）: 104403 (0.651), 104937 (0.664), 10203 (0.725)
- Sadj 下位（高い順）: 105490 (0.843), 10037 (0.760), 107280 (0.743)

### DISSE / elevation

- QNX(2) 上位: 10037 (0.743), 104403 (0.734), 105490 (0.708)
- QNX(2) 下位: 10203 (0.629), 106039 (0.667), 107280 (0.673)
- Sadj 上位（低い順）: 104937 (0.571), 104403 (0.638), 106039 (0.703)
- Sadj 下位（高い順）: 105490 (0.795), 10037 (0.758), 106612 (0.743)

### ELSA / azimuth

- QNX(2) 上位: 103289 (0.557), 10037 (0.552), 106612 (0.503)
- QNX(2) 下位: 106039 (0.441), 105490 (0.458), 103846 (0.481)
- Sadj 上位（低い順）: 104937 (0.410), 104403 (0.436), 106612 (0.577)
- Sadj 下位（高い順）: 103289 (0.702), 10037 (0.698), 105490 (0.668)

### ELSA / elevation

- QNX(2) 上位: 106612 (0.816), 104403 (0.813), 10037 (0.798)
- QNX(2) 下位: 106039 (0.535), 107280 (0.661), 10203 (0.684)
- Sadj 上位（低い順）: 104403 (0.147), 104937 (0.229), 106612 (0.333)
- Sadj 下位（高い順）: 105490 (0.665), 10037 (0.617), 10203 (0.589)

## 音源IDとAudioCapsイベント

音源IDだけではイベント内容を解釈できないため、AudioCapsのcaptionを併記する。以下のcaptionは実験で使用した音源IDとの対応確認用であり、カテゴリ効果の統計的検定ではない。

| source | caption |
|---|---|
| 10037 | A woman gives a speech and a crowd claps |
| 10203 | Truck in idle mode, door closing, engine revving and accelerating |
| 103289 | A man speaks followed by a loud burst then laughter |
| 103846 | A man talks in the distance followed by several vehicles driving and accelerating |
| 104403 | Wood lightly shuffling as insects buzz and birds chirp in the background |
| 104937 | A toy motor buzzing as water splashes and birds chirp in the distance. |
| 105490 | An engine running and helicopter propellers spinning |
| 106039 | A train running and the horn honking and bells chiming |
| 106612 | Something jangles then someone begins speaking then a door clanks |
| 107280 | A woman talking as a baby talks followed by plastic thumping |

## 残響に対する音源イベント別傾向

傾きは、各部屋を平均した QNX(2) または Sadj を T30=0, 0.3, 0.6 s に対して近似した記述量である。QNX(2) は負の傾き、Sadj は正の傾きが残響による悪化方向を表す。

| model | axis | QNXが最も低下する音源 | QNXが最も安定する音源 | Sadjが最も悪化する音源 | Sadjが最も安定する音源 |
|---|---|---|---|---|---|
| DISSE | azimuth | 10203 (-0.378; Truck in idle mode, door closing, engine revving and) | 10037 (-0.139; A woman gives a speech and a crowd claps) | 106612 (+0.583; Something jangles then someone begins speaking then ) | 105490 (-0.011; An engine running and helicopter propellers spinning) |
| DISSE | elevation | 106039 (-0.278; A train running and the horn honking and bells chimi) | 103289 (+0.029; A man speaks followed by a loud burst then laughter) | 103846 (+0.592; A man talks in the distance followed by several vehi) | 104937 (+0.067; A toy motor buzzing as water splashes and birds chir) |
| ELSA | azimuth | 103289 (-0.328; A man speaks followed by a loud burst then laughter) | 106612 (-0.054; Something jangles then someone begins speaking then ) | 106612 (+0.893; Something jangles then someone begins speaking then ) | 104937 (-0.061; A toy motor buzzing as water splashes and birds chir) |
| ELSA | elevation | 106039 (-0.994; A train running and the horn honking and bells chimi) | 106612 (-0.365; Something jangles then someone begins speaking then ) | 10037 (+1.196; A woman gives a speech and a crowd claps) | 104937 (-0.079; A toy motor buzzing as water splashes and birds chir) |

## DISSE と ELSA の音源別比較

QNX(2) は DISSE−ELSA が正なら DISSE が高く、Sadj は DISSE−ELSA が負なら DISSE が低く（良く）なる。次の比較は9条件平均である。

| axis | source | ΔQNX (DISSE−ELSA) | ΔSadj (DISSE−ELSA) | DISSEがQNXで優位な条件割合 | DISSEがSadjで優位な条件割合 |
|---|---|---:|---:|---:|---:|
| azimuth | 10037 | 0.085 | 0.063 | 0.89 | 0.67 |
| azimuth | 10203 | 0.139 | 0.095 | 1.00 | 0.00 |
| azimuth | 103289 | 0.102 | 0.024 | 1.00 | 0.11 |
| azimuth | 103846 | 0.164 | 0.080 | 1.00 | 0.22 |
| azimuth | 104403 | 0.181 | 0.215 | 1.00 | 0.00 |
| azimuth | 104937 | 0.197 | 0.254 | 1.00 | 0.00 |
| azimuth | 105490 | 0.167 | 0.175 | 1.00 | 0.11 |
| azimuth | 106039 | 0.161 | 0.116 | 1.00 | 0.00 |
| azimuth | 106612 | 0.146 | 0.154 | 0.89 | 0.00 |
| azimuth | 107280 | 0.156 | 0.085 | 1.00 | 0.11 |
| elevation | 10037 | -0.056 | 0.141 | 0.33 | 0.33 |
| elevation | 10203 | -0.056 | 0.128 | 0.33 | 0.00 |
| elevation | 103289 | -0.041 | 0.266 | 0.44 | 0.00 |
| elevation | 103846 | -0.012 | 0.210 | 0.56 | 0.11 |
| elevation | 104403 | -0.079 | 0.490 | 0.33 | 0.00 |
| elevation | 104937 | 0.012 | 0.342 | 0.56 | 0.00 |
| elevation | 105490 | -0.006 | 0.130 | 0.22 | 0.11 |
| elevation | 106039 | 0.132 | 0.238 | 0.67 | 0.00 |
| elevation | 106612 | -0.120 | 0.410 | 0.11 | 0.00 |
| elevation | 107280 | 0.012 | 0.244 | 0.56 | 0.11 |

## 解釈

- 方位角では、DISSE の QNX(2) は10音源すべてで ELSA より高い。一方、Sadj は ELSA の方が低い音源が多く、近傍の保存と隣接距離の均一性は一致しない。
- 仰角では、音源によって QNX(2) のモデル優位性が変わる。したがって、仰角については10音源平均だけで DISSE の一貫した優位性を主張せず、音源イベント別のばらつきを併記すべきである。
- 音源IDごとの差は、音源内容そのものの影響と、収録音量・無音区間・音源長などの前処理差の影響を含む。IDだけから「人の声」「環境音」などの意味カテゴリを断定するには、AudioCapsのキャプション等との対応付けが必要である。
- 本結果は記述統計であり、10音源だけで母集団全体への統計的有意性を主張しない。論文では音源をランダム効果とする解析、または音源ID bootstrapを追加する。

## 推奨する論文での示し方

1. 全体平均表に加えて、音源ID×モデル×軸のヒートマップを補足図にする。
2. 本文では各軸のQNX(2)上位・下位音源を1〜2例だけ示し、全10音源表は補足表に置く。
3. 音源内容による説明を行う場合は、音源IDとキャプション／カテゴリを結合し、カテゴリ別平均を別途計算する。
