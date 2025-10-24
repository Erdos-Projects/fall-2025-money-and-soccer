#!/bin/bash
echo "Reproducing final results..."
jupyter nbconvert --to notebook --execute notebooks/final_results.ipynb
echo "Artifacts and plots saved under results/final/"