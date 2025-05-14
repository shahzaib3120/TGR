# TGR Attack Evaluation Framework

This README provides a structured guide for running experiments with the TGR (Token Gradient Regularization) attack framework, collecting results, and analyzing them for research purposes.

## Setup

1. Make sure you've installed all required dependencies:

   ```bash
   pip install torch torchvision pandas matplotlib seaborn scikit-learn
   ```

2. Ensure you have the following files in your working directory:
   - attack.py - Generates adversarial examples
   - evaluate.py - Evaluates transferability to other models
   - `results_logger.py` - New file for logging results
   - `analyze_results.py` - New file for analyzing and visualizing results
   - `analyze_transferability.py` - New file for analyzing transferability

## Experiment Pipeline

### 1. Generate Adversarial Examples

Generate adversarial examples using different models as sources:

```bash
# ViT models
python attack.py --attack TGR --model_name vit_base_patch16_224 --batch_size 20
python attack.py --attack TGR --model_name deit_base_distilled_patch16_224 --batch_size 20
python attack.py --attack TGR --model_name cait_s24_224 --batch_size 20
python attack.py --attack TGR --model_name pit_b_224 --batch_size 20

# Try different epsilon values
python attack.py --attack TGR --model_name vit_base_patch16_224 --batch_size 20 --epsilon 0.03
python attack.py --attack TGR --model_name vit_base_patch16_224 --batch_size 20 --epsilon 0.05
```

### 2. Evaluate Transferability

Evaluate the transferability of the generated adversarial examples across different models:

```bash
# Evaluate examples from vit_base_patch16_224 on other models
python evaluate.py --adv_path model_vit_base_patch16_224-method_TGR --model_name deit_base_distilled_patch16_224
python evaluate.py --adv_path model_vit_base_patch16_224-method_TGR --model_name levit_256
python evaluate.py --adv_path model_vit_base_patch16_224-method_TGR --model_name pit_b_224
python evaluate.py --adv_path model_vit_base_patch16_224-method_TGR --model_name cait_s24_224

# Evaluate examples from other models
python evaluate.py --adv_path model_deit_base_distilled_patch16_224-method_TGR --model_name vit_base_patch16_224
```

For convenience, you can use the provided evaluation script to evaluate on all models:

```bash
bash run_evaluate.sh model_vit_base_patch16_224-method_TGR
```

### 3. Analyze Results

Once you've generated and evaluated multiple sets of adversarial examples, analyze the results:

```bash
# Analyze attack results
python analyze_results.py --log_dir results_logs --output_dir analysis_results

# Analyze transferability specifically
python analyze_transferability.py --log_dir evaluation_logs --output_dir transferability_results
```

## Experiment Ideas

Here are specific experiments you can perform to generate results for your research paper:

### 1. Cross-Architecture Transferability Study

**Goal**: Investigate how well TGR attacks transfer across different vision transformer architectures.

**Procedure**:

1. Generate adversarial examples using each model as the source
2. Evaluate each set of examples on all other models
3. Generate a transferability heatmap

```bash
# Generate examples from each model
python attack.py --attack TGR --model_name vit_base_patch16_224
python attack.py --attack TGR --model_name deit_base_distilled_patch16_224
python attack.py --attack TGR --model_name pit_b_224
python attack.py --attack TGR --model_name cait_s24_224

# Run evaluation for all combinations
bash run_evaluate.sh model_vit_base_patch16_224-method_TGR
bash run_evaluate.sh model_deit_base_distilled_patch16_224-method_TGR
bash run_evaluate.sh model_pit_b_224-method_TGR
bash run_evaluate.sh model_cait_s24_224-method_TGR

# Analyze transferability
python analyze_transferability.py
```

### 2. Epsilon Parameter Study

**Goal**: Determine how the perturbation magnitude (epsilon) affects attack success rate and transferability.

**Procedure**:

1. Generate examples using different epsilon values
2. Evaluate success rates and transferability
3. Create plots showing the relationship between epsilon and success rate

```bash
# Generate examples with different epsilon values
python attack.py --attack TGR --model_name vit_base_patch16_224 --epsilon 0.01
python attack.py --attack TGR --model_name vit_base_patch16_224 --epsilon 0.03
python attack.py --attack TGR --model_name vit_base_patch16_224 --epsilon 0.05
python attack.py --attack TGR --model_name vit_base_patch16_224 --epsilon 0.07
python attack.py --attack TGR --model_name vit_base_patch16_224 --epsilon 0.1

# Evaluate each set
for eps in 0.01 0.03 0.05 0.07 0.1; do
  bash run_evaluate.sh model_vit_base_patch16_224-method_TGR_eps_${eps}
done

# Analyze results
python analyze_results.py --plot_type all
```

### 3. CNN vs. Vision Transformer Robustness Comparison

**Goal**: Compare the vulnerability of CNNs versus Vision Transformers to TGR attacks.

**Procedure**:

1. Generate adversarial examples from different model types
2. Evaluate on both CNN models and Vision Transformer models
3. Compare success rates

```bash
# Generate examples
python attack.py --attack TGR --model_name vit_base_patch16_224

# Evaluate on ViT models
bash run_evaluate.sh model_vit_base_patch16_224-method_TGR

# Evaluate on CNN models
python evaluate_cnn.py --adv_path model_vit_base_patch16_224-method_TGR

# Analyze the results
python analyze_cnn_vs_vit.py  # You would need to create this script
```

### 4. Attack Steps Analysis

**Goal**: Determine the optimal number of attack steps for the TGR attack.

**Procedure**:

1. Generate examples using different numbers of steps
2. Compare success rates and computational efficiency

```bash
# Generate examples with different numbers of steps
python attack.py --attack TGR --model_name vit_base_patch16_224 --steps 5
python attack.py --attack TGR --model_name vit_base_patch16_224 --steps 10
python attack.py --attack TGR --model_name vit_base_patch16_224 --steps 20
python attack.py --attack TGR --model_name vit_base_patch16_224 --steps 50

# Evaluate and analyze
python analyze_results.py
```

## Result Analysis and Visualization

After running your experiments, you can generate publication-quality visualizations:

1. Success rate comparison across models:

   ```bash
   python analyze_results.py --plot_type success_rate
   ```

2. Transferability heatmap:

   ```bash
   python analyze_transferability.py
   ```

3. Perturbation analysis:

   ```bash
   python analyze_results.py --plot_type perturbation
   ```

4. Generate tables for your paper:
   ```bash
   python analyze_results.py  # This will generate CSV and LaTeX tables
   ```

## Research Paper Structure

Based on your experimental results, consider the following structure for your research paper:

1. **Introduction**

   - Background on adversarial attacks
   - Introduction to Vision Transformers
   - Importance of transferability

2. **Methodology**

   - Description of TGR attack algorithm
   - Experimental setup and model details
   - Evaluation metrics

3. **Results**

   - Attack success rates on different architectures
   - Transferability analysis
   - Parameter sensitivity study

4. **Analysis**

   - CNN vs. ViT robustness comparison
   - Architecture-specific vulnerabilities
   - Implications for model security

5. **Conclusion**
   - Summary of findings
   - Potential defenses
   - Future work

Your results and visualizations from the above experiments will provide the evidence needed to support your findings and conclusions.
