import torch
def sample_loss_landscape(model, loss_fn, data, target, epsilon=1e-3):

    original_params = []

    for p in model.parameters(): # model.parameters() returns an iterator over all the parameters of the model
        original_params.append(p.data.clone()) # clone the data of the parameter and store it in original_params

        noise = []

        for p in model.parameters():
            noise.append(torch.rand_like(p) * epsilon) # generate random noise of the same shape as the parameter and scale it by epsilon

        for p, n in zip(model.parameters(), noise):
            p.data += n
        output = model(data)

        perturbed_loss = loss_fn(output, target).item()

        for p, orig in zip(model.parameters(), original_params):
            p.data = orig
        return perturbed_loss
    
