## Paper

This is the code for the paper [Information-Theoretic Progress Measures reveal Grokking is an Emergent Phase Transition](https://openreview.net/pdf?id=Q4NH6hEPIX) by Kenzo Clauw, Daniele Marinazzo, and Sebastiano Stramaglia


# Assumptions
- Data is stored locally
- Logging via Neptune
- You have to use Jax 0.3.25

# Train Model
python src/main.py experiment=exp1_modular_add_adamw_weight_decay_euclidean_longer set_cluster_folder=False gridsearch_enabled=True train_enabled=True oinfo_enabled=False acc_enabled=False lth_enabled=False n_jobs_pool=4 ignore_existing_files=True device=cuda

# Run Oinfo
python src/main.py experiment=exp1_modular_add_adamw_weight_decay_euclidean_longer set_cluster_folder=False device=cuda gridsearch_enabled=False train_enabled=False oinfo_enabled=True acc_enabled=False lth_enabled=False ignore_existing_files=False root_folder_type=local neptune.enabled=False

# Run Lottery Ticket Hypothesis
python src/main.py experiment=exp1_modular_add_adamw_weight_decay_euclidean_longer set_cluster_folder=False gridsearch_enabled=False lth_enabled=True train_enabled=False oinfo_enabled=False acc_enabled=False n_jobs_pool=8 overwrite=True root_folder_type=local neptune.enabled=False

# Install jaxlib on GPU via the wheel archive
pip install jax[gpu]==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install the jaxlib 0.3.25 GPU wheel directly
pip install jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html



