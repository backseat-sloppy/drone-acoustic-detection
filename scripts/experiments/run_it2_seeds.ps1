# Iteration 2: repeat the two leave-one-chain-out runs with other seeds.
$py = ".\.venv-dml\Scripts\python.exe"
foreach ($seed in 1, 2) {
    & $py -u scripts\run_real.py --name "it2_ddl_near_s$seed" --models dscnn --variants buried --positives ddl-near --device dml --windows 200000 --seed $seed
    & $py -u scripts\run_real.py --name "it2_dronenoise_s$seed" --models dscnn --variants buried --positives dronenoise --dn-holdout none --device dml --windows 200000 --seed $seed
}
