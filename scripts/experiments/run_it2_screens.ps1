# Iteration 2, entry 016 step 3: one-seed screens of four data changes, both leave-one-chain-out arms.
# Arm X: train on DDL near, unseen chain DroneNoise. Arm Y: train on DroneNoise, unseen chain DDL.
$py = ".\.venv-dml\Scripts\python.exe"
$common = @("--variants", "buried", "--device", "dml", "--windows", "200000")
$x = @("--positives", "ddl-near")
$y = @("--positives", "dronenoise", "--dn-holdout", "none")

& $py -u scripts\run_real.py --name scr_base_lgbm_x --models lgbm @x @common
& $py -u scripts\run_real.py --name scr_base_lgbm_y --models lgbm @y @common
& $py -u scripts\run_real.py --name scr_near75_x  --models lgbm dscnn @x --near-m 75 @common
& $py -u scripts\run_real.py --name scr_near100_x --models lgbm dscnn @x --near-m 100 @common
& $py -u scripts\run_real.py --name scr_steady_x  --models lgbm dscnn @x --steady-boost 3 @common
& $py -u scripts\run_real.py --name scr_steady_y  --models lgbm dscnn @y --steady-boost 3 @common
& $py -u scripts\run_real.py --name scr_snr_x     --models lgbm dscnn @x --snr -10 15 @common
& $py -u scripts\run_real.py --name scr_snr_y     --models lgbm dscnn @y --snr -10 15 @common
