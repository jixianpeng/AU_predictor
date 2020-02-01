from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet

def load_model():
    # model = MobileNetV2_unet(None).to(args.device)
    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    # state_dict = torch.load(args.pre_trained, map_location='cpu')
    state_dict = torch.load('./checkpoints/model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


# image_files = sorted(glob('{}/*.jp*g'.format(args.data_folder)))
# image_files = sorted(glob('{}/*.jp*g'.format('D:\DATASET/aff_wild2/adversial_samples/')))
model = load_model()
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

def seg(image_files):
    images=[]
    raw_image=[]
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        images.append(torch_img)
        # image = np.asarray(Image.fromarray(image).convert('L'), dtype=np.uint8)/256.0
        image = np.asarray(Image.fromarray(image).convert('L'), dtype=np.uint8)
        raw_image.append(image)
    raw_image=np.asarray(raw_image)
    torch_img=torch.cat(images,dim=0)
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)
    raw_image[np.where(mask != 1)[0],np.where(mask != 1)[1] // 2, np.where(mask != 1)[2] // 2, ] = 0.0
    return raw_image


