# Iteration 5, entry 036: seeds 1 and 2 of the pooled screen (small CNN), with and without speed perturbation.
$py = ".\.venv-dml\Scripts\python.exe"
$arms = "nightjar", "d3set", "fpv:ariel_25.02", "fpv:palmahim_10.02", "fpv:palmahim_27.05"
foreach ($seed in 1, 2) {
    & $py -u scripts\run_range.py --name "it5_pool_s$seed" --pool --arms $arms --recipes simdist --models dscnn --device dml --windows 200000 --integrate 1 2 --seed $seed
    & $py -u scripts\run_range.py --name "it5_pool_speed_s$seed" --pool --arms $arms --recipes simdist --models dscnn --device dml --windows 200000 --integrate 1 2 --speed 0.8 1.25 --seed $seed
}
