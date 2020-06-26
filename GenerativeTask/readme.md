### run
- VAE
    ```./vae_img``` records reconstruction results every 10 epochs
    ```./saved_path``` saves the trained model
    ```./flow_img``` saves interpolation results of latent vector
    > python VAE_mnist.py

- t-SNE
    ```./t-SNE_result``` saves t-SNE visualization(2D, 3D) of different layers with/without PCA. 
    preprocess to get latent vector
    > python t-SNE_preprocess.py
    
    t-SNE main process 
    > python t-SNE.py

- PCA
    ```./PCA_compare``` saves PCA results of different layers.
    preprocess is same as t-SNE.
    > python PCA_clustering.py

