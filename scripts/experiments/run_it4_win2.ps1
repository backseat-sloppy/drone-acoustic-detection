# Iteration 4, step 2: the same two recipes trained and scored on 2 s windows, seed 0,
# with 1, 2 and 4 consecutive 2 s windows averaged.
$py = ".\.venv-dml\Scripts\python.exe"
& $py -u scripts\run_range.py --name it4_win2 --arms nightjar d3set --recipes near simdist --models dscnn --device dml --windows 200000 --window-seconds 2 --integrate 1 2 4
