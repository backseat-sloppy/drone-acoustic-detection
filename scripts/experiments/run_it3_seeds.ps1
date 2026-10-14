# Iteration 3, entry 028: comb rescored with the tie-safe threshold, and seeds 1 and 2
# of the near and simdist recipes, both arms, small CNN.
$py = ".\.venv-dml\Scripts\python.exe"
& $py -u scripts\run_range.py --name it3_comb --arms nightjar d3set --recipes near --models comb --device dml
foreach ($seed in 1, 2) {
    & $py -u scripts\run_range.py --name "it3_seed$seed" --arms nightjar d3set --recipes near simdist --models dscnn --device dml --windows 200000 --seed $seed
}
