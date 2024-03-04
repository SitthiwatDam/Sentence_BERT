# A5 Sentence Embedding with BERT
 

## Custom-Trained Sentence Transformer Training Process
The complete training process is detailed within the `A5_st123994.ipynb` notebook, providing in-depth instructions. Feel free to study and build upon this comprehensive guide for a better understanding of the custom-trained sentence transformer training.


## Web application with Flask
This web application is built using Flask, a popular web framework in Python. Using Flask, the application integrates a custom-trained sentence Transformer model from the training phase. The saved model is imported and loaded into the BERT class within this application, which is structured similarly to the one in the Jupyter notebook. All functionalities are then executed using the same functions and parameters employed during the training phases.

### Quick Start with Docker Compose

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SitthiwatDam/A5_Sentence_BERT.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd A5_Sentence_BERT
    ```

3. **Build and run the Docker containers:**
    ```bash
    docker-compose up -d
    ```

4. **Access the application:**
    - Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

5. **Submit a pair of sentence:**
    - Enter a sentence in the text areas.
    - Click the "Calculate" button.
    - Cosine Similarity value will show below.

6. **Stop the application when done:**
    ```bash
    docker-compose down
    ```

### Web Application Interface
![Web application interface](./pics/picture1.png)



