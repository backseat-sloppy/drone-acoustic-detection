# Iteration 4, step 1: rescore the iteration 3 checkpoints (near and simdist, three seeds,
# both arms) with the scores of k consecutive 1 s windows averaged, k = 1, 2, 4, 8.
$py = ".\.venv-dml\Scripts\python.exe"
$seed = 0
foreach ($run in "it3_screen", "it3_seed1", "it3_seed2") {
    & $py -u scripts\run_range.py --name "it4_int_$run" --load-from $run --arms nightjar d3set --recipes near simdist --models dscnn --device dml --integrate 1 2 4 8 --seed $seed
    $seed += 1
}
