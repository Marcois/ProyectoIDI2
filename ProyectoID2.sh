#!/bin/bash

echo "Select an option:"
echo "1) Full loop (build dataset → train → evaluate)"
echo "2) Build processed dataset"
echo "3) Train model"
echo "4) Evaluate model"
echo -n "Enter your choice (1-4): "
read choice

run_step() {
    script_name=$1
    description=$2

    echo "Running $script_name ($description)..."
    start=$(date +%s)

    python "$script_name"

    end=$(date +%s)
    echo "$script_name finished in $((end - start)) seconds."
    echo
}

case "$choice" in
    1)
        echo "Executing full pipeline..."
        run_step "build_processed_datasets.py" "building processed dataset"
        run_step "train_models.py" "training model"
        run_step "models_evaluation.py" "evaluating model"
        ;;
    2)
        run_step "build_processed_datasets.py" "building processed dataset"
        ;;
    3)
        run_step "train_models.py" "training model"
        ;;
    4)
        run_step "models_evaluation.py" "evaluating model"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo "--------------------------------------------"
echo "Execution of pipeline done."