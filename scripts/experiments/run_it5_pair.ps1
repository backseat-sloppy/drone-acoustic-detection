# Iteration 5, entry 036: the seed 0 pooled checkpoints rescored on two-microphone averages (no training).
$py = ".\.venv-dml\Scripts\python.exe"
& $py -u scripts\run_range.py --name it5_pair_pool --load-from it5_pool --pool --pair --arms nightjar d3set --recipes simdist --models dscnn --device dml --integrate 1 2
& $py -u scripts\run_range.py --name it5_pair_pool_speed --load-from it5_pool_speed --pool --pair --arms nightjar d3set --recipes simdist --models dscnn --device dml --integrate 1 2
