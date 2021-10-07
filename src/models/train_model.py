import torch 
import torchvision
from src.nets.rmac_resnet import resnet101_rmac
from src.data.dataset import ImageList
from torch.utils.data import DataLoader
from src.models.loss import APLoss
import matplotlib.pyplot as plt
def train () :
	pass

def evaluate() :
	pass

if __name__ =="__main__":
	#load images
	transfroms = torchvision.transforms.ToTensor()
	imageList = [line.strip() for line in open("/homes/bacharya/deep-image-retrieval/dirtorch/dataset/list_queries")]
	queryImages = ImageList(imageList, transform=transfroms)
	model = resnet101_rmac()
	with torch.no_grad():
		dList = list()
		for i in range(10):
			print(i)
			vector = model(torch.unsqueeze(queryImages[i], dim=0))
			dList.append(vector.unsqueeze(0).cpu())
			del vector
			torch.cuda.empty_cache()
	D = torch.cat(dList,0)
	#calculate the similarity matrix
	D.requires_grad = True
	simD = D@D.T
	#dummy y
	y = torch.randint(0,2,size=simD.shape)
	loss = APLoss(min = -1)
	mAp = loss(simD, y)
	mAp.backward()
	print(D.grad)
	print(D.shape)
	
# queryLoader = DataLoader(queryImages, batch_size=1, shuffle=False)
