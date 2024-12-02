# Nature's Remedy

This is a website for identification of medicinal plants and discovering their properties. The website has two modes.  

  
In Plant Mode, users can upload a picture of a medicinal plant, which if recognized will return a list of its properties. 
This has been implemented using a ResNet50 model finetuned on a kaggle dataset of 40 medicinal plants. Refer to medicinalplantresnet50.ipynb for details about the model.  

  
In Problem Mode, users can input their health problems and discover medicinal plant-based remedies that can help. 
These results are fetched from a mongoDB database containing the plants, their properties, problems solved and the uses. 
This information can be found in the plantData.json file.

Webpage Samples:  

Home Page  

![HomeSample](https://github.com/user-attachments/assets/66b17065-ce0c-4e79-9156-9a3f37ea74f4)  


Plant Mode Results  

![PlantmodeSample](https://github.com/user-attachments/assets/d8729548-9700-4b55-9213-907e04a2e739)  


Problem Mode Results  

![ProblemmodeSample](https://github.com/user-attachments/assets/ca31dd24-4d60-490a-933b-1bef8b85d243)
