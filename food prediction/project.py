import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd

classes = ['pizza','steak','sushi']
classid = {0:'pizza',1:'steak',2:'sushi'}
class_desc = {
    0:'pizza, dish of Italian origin consisting of a flattened disk of bread \n dough topped with some combination of olive oil, oregano, tomato, olives, mozzarella or other cheese, and many other ingredients, baked quickly—usually, in a commercial setting, using a wood-fired oven heated to a very high temperature—and served hot ...',
    1:'A steak is a thick cut of meat generally sliced across the muscle fibers,\n sometimes including a bone. It is normally grilled or fried. Steak can be diced, cooked in sauce, such as in steak and kidney pie, or minced and formed into patties, such as hamburgers.',
    2:'sushi, a staple rice dish of Japanese cuisine, consisting of cooked rice \nflavoured with vinegar and a variety of vegetable, egg, or raw seafood garnishes and served cold.'
}

df = pd.read_csv("nutrients.csv")

from torch.nn.modules.conv import Conv2d
class FoodClassification(nn.Module):
  def __init__(self,input_shape=3,output_shape=len(classes)):
    super().__init__()

    self.Conv_blk_1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=2,stride=1),
        nn.ReLU(),
        #nn.Dropout2d(0.5),
        nn.MaxPool2d(kernel_size=2,stride=1),
        nn.Conv2d(64,32,kernel_size=2,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.avgpool = nn.AvgPool2d(kernel_size=2)
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(32*15*15,32),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(32,32),
        nn.ReLU(),
        nn.Linear(32,output_shape)
    )
  def forward(self, x):
        x = self.Conv_blk_1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
  

torch.manual_seed(42)
model = FoodClassification()

model.load_state_dict(torch.load('trained Models/prototype12.pth',map_location=torch.device('cpu')))

transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

def project_model():
   img_loc = input("enter the location of image")
   img = Image.open(img_loc)

   transformed = transform(img)

   model.eval()
   with torch.no_grad():
     y_pred = model(transformed.unsqueeze(0))
     print(f"prediction logits:{y_pred}")
     y_pred = torch.softmax(y_pred,dim=1)
     print(f"prediction probability:{y_pred}")
     y_pred = torch.argmax(y_pred,dim=1)
     y_pred = y_pred.data.item()
     print(f"predicted image:{classid[y_pred]}")
     print(f"Description:\n{class_desc[y_pred]}")
     print(f"nutrition content:\n{df.iloc[y_pred]}")


project_model()