from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_gradcam(model, input_tensor, target_category):
    cam = GradCAM(model=model, target_layers=[model.backbone.layer4[-1]], use_cuda=device.type == "cuda")
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=None)[0]
    image = input_tensor.permute(1, 2, 0).cpu().numpy()
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.title(f"GradCAM - Target class: {target_category}")
    plt.axis('off')
    plt.show()


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embeddings(model, dataloader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            feats = model.backbone.avgpool(model.backbone.layer4(inputs))
            features.extend(feats.squeeze().cpu().numpy())
            labels.extend(lbls.numpy())

    tsne = TSNE(n_components=2).fit_transform(features)
    plt.scatter(tsne[:,0], tsne[:,1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.title("t-SNE Visualization")
    plt.show()
