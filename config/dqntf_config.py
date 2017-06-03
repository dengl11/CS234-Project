class DQNTfConfig:
	# output config
    output_path  = "results/qLearnNN/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"
    grad_clip    = False
    clip_val     = 10
    save_every   = 100 # in batch
    reg          = 1e-6