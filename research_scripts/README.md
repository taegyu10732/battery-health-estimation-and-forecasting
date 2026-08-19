# Consolidated research scripts

These 35 scripts preserve the architecture and methodology coverage of the 106 private source notebooks. Generated outputs, pinned GPU selection, and scratch-only cells were removed. Closely related runs are represented by one script; the final column records what it consolidates.

These are research records, not all verified end-to-end examples. Start with the four curated scripts under `workflows/` for the cleanest execution path.

| ID | Area | Consolidated script | Architecture | Method tags | Covers |
| --- | --- | --- | --- | --- | --- |
| G01 | clustering | [Engineered-feature lifetime clustering](clustering/01_engineered_feature_mlp.py) | MLP | engineered features|lifetime classes | nature feature variants |
| G02 | clustering | [Raw-curve lifetime clustering](clustering/02_raw_curve_densenet.py) | DenseNet CNN | raw curves|classification | raw data|changed data |
| G03 | soh | [Normalized SOH baseline](soh/01_transformer_cnn_baseline.py) | CNN-Transformer encoder | Optuna|weighted loss|input ablation | 90|120|180|240|300 cycles|no temperature|charge only|voltage only|rare|tester |
| G04 | soh | [SOH MLP baseline](soh/02_mlp_baseline.py) | MLP | Optuna|SOH regression | MLP baseline |
| G05 | soh | [Transformer fine tuning](soh/03_transformer_fine_tuning.py) | CNN-Transformer encoder | fine tuning|attention | fine tuning branch |
| G06 | soh | [Variational SOH model](soh/04_vae.py) | VAE CNN-Transformer | VAE|reconstruction | base VAE |
| G07 | soh | [Cycle-conditioned VAE](soh/05_cycle_conditioned_vae.py) | VAE CNN-Transformer | cycle conditioning|weighted loss | normalized-cycle VAE|cycle-input VAE |
| G08 | soh | [DeepHPM SOH model](soh/06_deephpm.py) | CNN-Transformer + MLP + multihead attention | DeepHPM|physics loss|cycle input | 120|180|300|empirical|rare|full |
| G09 | soh | [Knee-point estimation](soh/07_knee_detection.py) | CNN-Transformer encoder | knee detection|scarce data | knee|knee scarce |
| G10 | soh | [Post-knee capacity prediction](soh/08_post_knee_lstm.py) | LSTM | post-knee|gradient regularization | full|scarce post-knee |
| G11 | soh | [Surrogate degradation modeling](soh/09_surrogate_custom_lstm.py) | LSTM + custom recurrent cell | surrogate|multi-cycle | surrogate branches |
| G12 | rul | [DenseNet RUL baseline](rul/01_densenet_baseline.py) | DenseNet CNN | RUL regression | DenseNet baseline |
| G13 | rul | [Normalized RUL Transformer](rul/02_transformer_baseline.py) | CNN-Transformer encoder | Optuna|normalized inputs | 30|100 cycle baseline|true SOH |
| G14 | rul | [CNN-LSTM RUL model](rul/03_cnn_lstm.py) | CNN-LSTM | sequence window|operating profile | LSTM RUL branch |
| G15 | rul | [Operating-profile cycle-gradient model](rul/04_operating_profile_cycle_gradient.py) | CNN-Transformer encoder | operating profile|cycle gradient|adaptive loss | batch1|cycle gradient|OP |
| G16 | rul | [Dual-task OP and gradient model](rul/05_dual_task.py) | CNN-Transformer encoder | dual task|SOH auxiliary|cycle gradient | dual-task branch |
| G17 | rul | [No-OP ablation](rul/06_no_operating_profile_ablation.py) | CNN-Transformer encoder | operating-profile ablation | with OP|without OP |
| G18 | rul | [Multi-cycle additive fusion](rul/07_multi_cycle_plus.py) | CNN-Transformer encoder | multi-cycle|additive fusion | plus fusion |
| G19 | rul | [Multi-cycle SOH fusion](rul/08_multi_cycle_soh_fusion.py) | CNN-Transformer encoder | multi-cycle|SOH auxiliary | normal CNN|concat|SOH added |
| G20 | rul | [Weighted multi-cycle fusion](rul/09_multi_cycle_weighted.py) | CNN-Transformer encoder | multi-cycle|learned weighting | weighted plus|70-cycle |
| G21 | rul | [Fixed-window subgroup model](rul/10_fixed_window_subgroups.py) | CNN-Transformer encoder | 30-cycle window|life subgroup|SOH regularization | short|mid|long|sequence|true SOH|regularization |
| G22 | rul | [Knee-aware RUL model](rul/11_knee_aware.py) | CNN-Transformer encoder | knee point|cycle gradient | knee-aware RUL |
| G23 | sequence | [LSTM sequence-to-sequence](sequence/01_lstm_seq2seq.py) | LSTM encoder-decoder | beta weighting|extra information | beta 0.88|nature 100-to-100|extra information |
| G24 | sequence | [Masked teacher-forcing sequence model](sequence/02_masked_seq2seq.py) | LSTM encoder-decoder | masking|teacher forcing | masked|teacher forcing |
| G25 | sequence | [Dense-to-sequence model](sequence/03_cnn_gru_seq2seq.py) | DenseNet CNN + GRU decoder | raw encoder|GRU decoding | dense-to-sequence |
| G26 | physics | [DenseNet physics-loss comparison](physics/01_densenet_physics_loss.py) | DenseNet CNN | physics loss|data-only ablation | with physics|without physics |
| G27 | physics | [Fixed-parameter physics model](physics/02_fixed_parameter.py) | DenseNet CNN | fixed parameters|given parameters | fixed|extra input|separate|without first-cycle|copy |
| G28 | physics | [Additional-information physics model](physics/03_additional_information_ntk.py) | CNN-LSTM | additional information|100th cycle|NTK | additional info|100th|NTK |
| G29 | physics | [Constrained-parameter physics model](physics/04_parameter_constrained.py) | CNN-LSTM | parameter constraints|physics loss | constrained parameters |
| G30 | physics | [Light parameter-given model](physics/05_parameter_given_light.py) | CNN-LSTM | given parameters|light model | parameter-given light |
| G31 | physics | [Memory-parameter physics model](physics/06_memory_parameter.py) | CNN-LSTM + learnable memory | parameter memory|learnable parameters | memory-module variants |
| G32 | method development | [Pre-normalization cycle-SOH model](method_development/01_cycle_soh_transformer.py) | CNN-Transformer encoder | joint cycle and SOH | cycle estimation|identical physics copy |
| G33 | method development | [Ensemble Transformer Bayesian optimization](method_development/02_ensemble_transformer_bo.py) | CNN-Transformer encoder | ensemble|Optuna|observation horizon | 60|100|120|140|160|180|210|240|270|300|sampling |
| G34 | method development | [Pre-normalization physics estimator](method_development/03_pre_normalization_physics.py) | CNN-Transformer encoder | physics parameters|180-cycle | base physics|180-cycle |
| G35 | method development | [Empirical physics fusion](method_development/04_empirical_fusion.py) | CNN-Transformer encoder | empirical fusion|log-cycle | empirical fused|empirical log-cycle |
