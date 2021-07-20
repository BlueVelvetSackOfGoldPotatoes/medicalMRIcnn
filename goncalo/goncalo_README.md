# TODO TODAY
1. Feed UMCG images and check the output from the actual pipeline;
2. Understand what the output of the VGG16 is and how to integrate it into the pipeline;
3. Make the dataset - 100/100, umcg/ukbb;
4. Get the bottleneck layer to function state;

# GENERAL PLAN
1. Implement W. Bai, et al. 's  pipeline (particularly the short-axis model).
2. Test the performance of the model on a new dataset generated from a different site with a different machine protocol (UMCG). This should be done using the quantitative assessments also deployed in W. Bai, et al.
3. Perform transfer learning using UMCG data, more specifically the short-axis model that classifies RV on CMR images. Compare performance with that obtained in B. 
4. Monitor learning using live loss function plotting as enabled by tensorboard.

1. Calculate average vector from training data sample (PCA-prototype-vector or PCAPV) by using PCA to reduce each image and average the resulting vectors element-wise.
2. Repeat 2)A, Using model bottleneck layer instead of PCA for dimensionality reduction (model-knowledge-prototype-vector or MKPV).
3. Use PCAPV and MKPV to reduce a UMCG image.
4. Calculate similarity measure using Euclidean distance between UMCG reduced vectors and…
    A. PCAPV;
    B. MKPV.
5. Check differences between the results from D-a) and D-b).
6. Train the model on UMCG data and repeat A to E. Training strategy follows from the transfer-learning conducted in W. Bai, et al.: randomly splitting data into 80% subjects for fine-tuning and 20% for evaluation.
7. Compare the final results between vectors generated before training on UMCG data and after training.

### NOTES
Always go with the pipeline flow - it already does everything;

## WHAT I CAN DO NOW
Retrain network
	a) Make datasets
		a.i) 3,975/300/600 for training/validation/test

Fit PCAs
	a) Build pipeline
		a.i) get pngs (the same that were used to train the networks)
		a.ii) fit PCA
		a.iii) calculate distance measurement to bottleneck layer (the one that is the outcome of retraining the network)

