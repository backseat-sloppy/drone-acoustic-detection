# Iteration 2: rescore the six leave-one-chain-out models with the headline figure of entry 016.
$py = ".\.venv-dml\Scripts\python.exe"
foreach ($s in "", "_s1", "_s2") {
    & $py -u scripts\run_real.py --name "rescore_ddl_near$s" --load-from "it2_ddl_near$s" --models dscnn --variants buried --positives ddl-near --device dml
    & $py -u scripts\run_real.py --name "rescore_dronenoise$s" --load-from "it2_dronenoise$s" --models dscnn --variants buried --positives dronenoise --dn-holdout none --device dml
}
