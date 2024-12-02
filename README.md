# Nature's Remedy

This is a website for identification of medicinal plants and discovering their properties. The website has two modes.  

  
In Plant Mode, users can upload a picture of a medicinal plant, which if recognized will return a list of its properties. 
This has been implemented using a ResNet50 model finetuned on a kaggle dataset of 40 medicinal plants. Refer to medicinalplantresnet50.ipynb for details about the model.  

  
In Problem Mode, users can input their health problems and discover medicinal plant-based remedies that can help. 
These results are fetched from a mongoDB database containing the plants, their properties, problems solved and the uses. 
This information can be found in the plantData.json file.
