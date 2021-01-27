def log_parameter(model, args):
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    with open("param_size.txt", "a") as fd:
        fd.write("{} {} {}".format(args.net, args.model_size, pytorch_total_params))
        fd.write('\n')
        fd.close()
    print("complete")
    exit(-1)