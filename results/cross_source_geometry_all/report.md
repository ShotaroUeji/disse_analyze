# Cross-source geometry report

## 実行概要

実行コマンド: `/home/takamichi-lab-pc09/DELSA/cross_source_geometry/run_analysis.py --input-root results/continuity_additional_3x3_10sources --axis both --room all --t30 all --models elsa disse --experiments distance retrieval rsa --output-root results/cross_source_geometry_all`

保存済みのDISSE/ELSA埋め込みだけを読み込み、音声生成・RIR生成・モデル推論は行っていない。
方位角は周期的、仰角は非周期的な変数として扱った。45音源ペアは独立標本とみなさず、必要な要約では音源ID単位のbootstrapを使用する。

## 条件別の主要結果

| axis | room | T30 [s] | model | sources | dim | D_same | D_diff | Δ | Retrieval R@1 (LOSO) | RSA mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| azimuth | large | 0.0 | disse | 10 | 512 | 0.1962 | 0.4121 | 0.2160 | 0.3861 | 0.9756 |
| azimuth | large | 0.3 | disse | 10 | 512 | 0.2657 | 0.6239 | 0.3582 | 0.3403 | 0.9119 |
| azimuth | large | 0.6 | disse | 10 | 512 | 0.2697 | 0.6847 | 0.4150 | 0.3222 | 0.9229 |
| azimuth | medium | 0.0 | disse | 10 | 512 | 0.1962 | 0.4121 | 0.2160 | 0.3861 | 0.9756 |
| azimuth | medium | 0.3 | disse | 10 | 512 | 0.2204 | 0.6082 | 0.3878 | 0.3444 | 0.9150 |
| azimuth | medium | 0.6 | disse | 10 | 512 | 0.2275 | 0.6556 | 0.4281 | 0.3000 | 0.9293 |
| azimuth | small | 0.0 | disse | 10 | 512 | 0.1962 | 0.4121 | 0.2160 | 0.3861 | 0.9756 |
| azimuth | small | 0.3 | disse | 10 | 512 | 0.1628 | 0.5462 | 0.3834 | 0.4014 | 0.9321 |
| azimuth | small | 0.6 | disse | 10 | 512 | 0.2255 | 0.6188 | 0.3933 | 0.3542 | 0.9352 |
| elevation | large | 0.0 | disse | 10 | 512 | 0.2383 | 0.5377 | 0.2994 | 0.5842 | 0.9751 |
| elevation | large | 0.3 | disse | 10 | 512 | 0.2835 | 0.5839 | 0.3004 | 0.4579 | 0.9290 |
| elevation | large | 0.6 | disse | 10 | 512 | 0.2753 | 0.5666 | 0.2912 | 0.3632 | 0.9176 |
| elevation | medium | 0.0 | disse | 10 | 512 | 0.2383 | 0.5377 | 0.2994 | 0.5842 | 0.9751 |
| elevation | medium | 0.3 | disse | 10 | 512 | 0.2141 | 0.5008 | 0.2867 | 0.4105 | 0.9191 |
| elevation | medium | 0.6 | disse | 10 | 512 | 0.2276 | 0.5184 | 0.2908 | 0.3421 | 0.9285 |
| elevation | small | 0.0 | disse | 10 | 512 | 0.2383 | 0.5377 | 0.2994 | 0.5842 | 0.9751 |
| elevation | small | 0.3 | disse | 10 | 512 | 0.1529 | 0.4205 | 0.2676 | 0.5158 | 0.9418 |
| elevation | small | 0.6 | disse | 10 | 512 | 0.2122 | 0.4671 | 0.2549 | 0.5158 | 0.9640 |
| azimuth | large | 0.0 | elsa | 10 | 512 | 0.6954 | 0.7791 | 0.0837 | 0.1208 | 0.9768 |
| azimuth | large | 0.3 | elsa | 10 | 512 | 0.6484 | 0.7947 | 0.1463 | 0.1194 | 0.9180 |
| azimuth | large | 0.6 | elsa | 10 | 512 | 0.6468 | 0.8337 | 0.1869 | 0.1278 | 0.9188 |
| azimuth | medium | 0.0 | elsa | 10 | 512 | 0.6954 | 0.7791 | 0.0837 | 0.1208 | 0.9768 |
| azimuth | medium | 0.3 | elsa | 10 | 512 | 0.6268 | 0.7988 | 0.1719 | 0.1500 | 0.9107 |
| azimuth | medium | 0.6 | elsa | 10 | 512 | 0.6447 | 0.8497 | 0.2050 | 0.1444 | 0.9269 |
| azimuth | small | 0.0 | elsa | 10 | 512 | 0.6954 | 0.7791 | 0.0837 | 0.1208 | 0.9768 |
| azimuth | small | 0.3 | elsa | 10 | 512 | 0.6064 | 0.7874 | 0.1810 | 0.1431 | 0.9147 |
| azimuth | small | 0.6 | elsa | 10 | 512 | 0.6639 | 0.8470 | 0.1831 | 0.1819 | 0.9298 |
| elevation | large | 0.0 | elsa | 10 | 512 | 0.6994 | 0.7347 | 0.0353 | 0.1737 | 0.9529 |
| elevation | large | 0.3 | elsa | 10 | 512 | 0.6245 | 0.6653 | 0.0408 | 0.1579 | 0.8404 |
| elevation | large | 0.6 | elsa | 10 | 512 | 0.5988 | 0.6438 | 0.0450 | 0.2105 | 0.8240 |
| elevation | medium | 0.0 | elsa | 10 | 512 | 0.6994 | 0.7347 | 0.0353 | 0.1737 | 0.9529 |
| elevation | medium | 0.3 | elsa | 10 | 512 | 0.5900 | 0.6348 | 0.0448 | 0.1684 | 0.8248 |
| elevation | medium | 0.6 | elsa | 10 | 512 | 0.5792 | 0.6332 | 0.0540 | 0.1895 | 0.8446 |
| elevation | small | 0.0 | elsa | 10 | 512 | 0.6994 | 0.7347 | 0.0353 | 0.1737 | 0.9529 |
| elevation | small | 0.3 | elsa | 10 | 512 | 0.5504 | 0.6007 | 0.0503 | 0.2211 | 0.8475 |
| elevation | small | 0.6 | elsa | 10 | 512 | 0.5753 | 0.6369 | 0.0616 | 0.2211 | 0.8355 |

## 解釈上の注意

- D_same が小さいことは、同じ角度の音源間埋め込みが近いことを示す。
- RSA は音源内の距離構造の一致度であり、埋め込み座標の位置合わせそのものを証明しない。
- 高いRSAと大きいCross-source距離が同時に得られた場合、位置・向きのずれの可能性は示唆されるが、回転を断定しない。Procrustesは追加解析として扱う。
- Retrievalの正例は同一物理角度の別音源候補であり、chance levelを各CSVに併記した。

## 出力

条件別ディレクトリのCSVとPNG/PDF、`input_manifest.csv`、`aggregate_results.csv`を参照する。
