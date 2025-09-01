import torch.nn as nn
from anomalib.models.cflow_custom.mobilone import reparameterize_model,mobilone
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, List, Tuple,Dict
from torch import Tensor
import torch
import os
import anomalib.models.cflow_custom.fast_vit_models as fast_vit_models
from omegaconf import DictConfig, OmegaConf,open_dict
class FastVitFeatureExtractor(nn.Module):
    
    def __init__(self,):
        
        super().__init__()
        variant="fastvit_sa24"
        model = getattr(fast_vit_models, variant)(fork_feat=True)
        checkpoint= "/home/Develop/Models/fastvit/fastvit_sa24_reparam.pth.tar"
        
        self.feature_extractor = reparameterize_model(model)
        if checkpoint is not None:
            print(f"Load checkpoint {checkpoint}")
            chkpt = torch.load(checkpoint)
            self.feature_extractor.load_state_dict(chkpt["state_dict"],strict=False)
        
        
        self.feature_extractor.eval()
        
        self.out_dims =[]
        #This value could be anything beacause for the model the
        #important value is the number of channels, Height and Width could be any
        t=torch.rand(1,3,512,512)
        
        outs= self.feature_extractor(t)
        
        self.outputs={}
        
        for key in outs:
            
            print(key.size())
            self.out_dims.append(key.size()[1])
        
    
    def forward(self,x):    
        return self.feature_extractor(x)
        
    
    
class MobileOneFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.


    """

    def __init__(self, layers: List[str], pre_trained: bool = True, requires_grad: bool = False):
        super().__init__()

        # Extract backbone-name and weight-URI from the backbone string.
        model = mobilone(variant='s3')
        model.eval()      
        model_eval = reparameterize_model(model)
        
        #self.layers = OmegaConf.to_object(layers)
        #self.layers= ["stage1","stage2","stage3"]

        ##self.layers= ["stage3","stage4"] Otttimo risultato nella saldatura. Si concentra molto sull'immagine globale
        self.layers=OmegaConf.to_object(layers)
        
        base_path="/home/Develop/Models/"
        model_filename = "mobileone_s3.pth.tar"
        
        
        checkpoint = torch.load(os.path.join(base_path, model_filename))
        model_eval.load_state_dict(checkpoint)
        
        
        t=torch.rand(1,3,128,128)
        
        self.feature_extractor=create_feature_extractor(
            model_eval, return_nodes=self.layers)
        

        #self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        
        if not requires_grad:
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
        
        #Setup Output Dim
        self.out_dims =[]
        #This value could be anything beacause for the model the
        #important value is the number of channels, Height and Width could be any
       
        
        outs= self.feature_extractor(t)
        
        for key in outs:
            
            print(outs[key].size())
            self.out_dims.append(outs[key].size()[1])
        
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        # if self.requires_grad:
        #     features = dict(zip(self.layers, self.feature_extractor(inputs)))
        # else:
        #     self.feature_extractor.eval()
        #     with torch.no_grad():
        #         features = dict(zip(self.layers, self.feature_extractor(inputs)))
        # return features
        
        return self.feature_extractor(inputs)
    
    

if __name__ =="__main__":
   
    pass
    # feat= FastVitFeatureExtractor()
    
    # inp= torch.rand(1,3,256,256)
    
    # outs= feat(inp)
    
    # for out in outs:
        
    #     print(out.size())
    
    # model = mobilone(variant='s1')
    
    
    
    # model.eval()      
    # model_eval = reparameterize_model(model)
    
    
    # checkpoint = torch.load('/home/Develop/Models/mobilone/mobileone_s1.pth.tar')
    # model_eval.load_state_dict(checkpoint)
    
    # layers=["stage1","stage2","stage3"]
    # feature_extractor=create_feature_extractor(
    #         model_eval, return_nodes=layers)
    
    # t= torch.rand(1,3,224,224)
    
    # outs= feature_extractor(t)
    
    # for key in outs:
        
    #     print(outs[key].size())