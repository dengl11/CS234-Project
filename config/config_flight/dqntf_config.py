class DQNTfConfig:
    # output config
    output_path  = "results/qLearnNN/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"
    hidden_size_2= 120
    hidden_size  = 80
    grad_clip    = False
    clip_val     = 10
    save_every   = 100 # in batch
    reg          = 1e-4
    lr           = 8e-4
    beta         = 0.92