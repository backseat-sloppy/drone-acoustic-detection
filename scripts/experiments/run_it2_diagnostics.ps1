# Iteration 2 diagnostics: which training drones make the model hear the drone?
$py = ".\.venv-dml\Scripts\python.exe"
& $py -u scripts\run_real.py --name it2_ddl_near --models lgbm dscnn --variants buried --positives ddl-near --device dml --windows 200000
& $py -u scripts\run_real.py --name it2_dronenoise --models lgbm dscnn --variants buried --positives dronenoise --dn-holdout none --device dml --windows 200000
& $py -u scripts\run_real.py --name it2_dronenoise_no_m3 --models dscnn --variants buried --positives dronenoise --dn-holdout M3 --device dml --windows 200000
& $py -u scripts\run_real.py --name it2_both --models lgbm dscnn --variants buried --positives ddl-near dronenoise --dn-holdout M3 --device dml --windows 200000
