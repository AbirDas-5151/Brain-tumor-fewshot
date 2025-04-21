def error_analysis(model, dataloader, class_names):
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    img = inputs[i].permute(1,2,0).cpu().numpy()
                    plt.imshow(img)
                    plt.title(f"True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}")
                    plt.axis('off')
                    plt.show()
