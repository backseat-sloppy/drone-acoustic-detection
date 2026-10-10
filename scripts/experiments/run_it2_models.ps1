# Iteration 2, entry 016 step 5: bounded model screen, one seed, both leave-one-chain-out arms.
# Also one frontend knob that REQUIREMENTS.md leaves free: a 2 s window.
$py = ".\.venv-dml\Scripts\python.exe"
$common = @("--variants", "buried", "--device", "dml", "--windows", "200000")
$x = @("--positives", "ddl-near")
$y = @("--positives", "dronenoise", "--dn-holdout", "none")

& $py -u scripts\run_real.py --name mod_resnet_x --models resnet @x @common
& $py -u scripts\run_real.py --name mod_resnet_y --models resnet @y @common
& $py -u scripts\run_real.py --name mod_transformer_x --models transformer @x @common
& $py -u scripts\run_real.py --name mod_transformer_y --models transformer @y @common
& $py -u scripts\run_real.py --name win2s_x --models lgbm dscnn @x @common --window-seconds 2.0
& $py -u scripts\run_real.py --name win2s_y --models lgbm dscnn @y @common --window-seconds 2.0
