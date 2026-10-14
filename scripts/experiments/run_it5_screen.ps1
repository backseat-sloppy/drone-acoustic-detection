# Iteration 5, entry 034: pooled training, one unit held out at a time (entry 033), seed 0.
$py = ".\.venv-dml\Scripts\python.exe"
$arms = "nightjar", "d3set", "fpv:ariel_25.02", "fpv:palmahim_10.02", "fpv:palmahim_27.05"
& $py -u scripts\run_range.py --name it5_pool --pool --arms $arms --recipes simdist --models lgbm dscnn --device dml --windows 200000 --integrate 1 2
& $py -u scripts\run_range.py --name it5_pool_speed --pool --arms $arms --recipes simdist --models dscnn --device dml --windows 200000 --integrate 1 2 --speed 0.8 1.25
