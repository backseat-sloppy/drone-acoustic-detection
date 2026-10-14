# Iteration 3, entry 027: leave-one-rig-out screen of the three recipes of entry 025.
$py = ".\.venv-dml\Scripts\python.exe"
& $py -u scripts\run_range.py --name it3_screen --arms nightjar d3set --recipes near range simdist --models comb lgbm dscnn --device dml --windows 200000
