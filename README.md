# CS4782-Final-Project-LLMs-and-Arithmetic

<h2 dir="ltr">3.1 Introduction</h2> 
<ul dir="ltr"> 
In the paper "Limitations of Language Models in Arithmetic and Symbolic Induction," Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan from the University of California, Santa Barbara, examine how large pretrained language models like GPT-3 and T5 handle arithmetic and symbolic reasoning tasks, with a particular focus on addition. The research highlights that despite their strong performance, these models face difficulties with complex arithmetic that includes longer digits and repeated numbers. Despite various efforts, such as fine-tuning and the use of detailed computation steps and positional markers, these models still struggle with out-of-distribution data and repeating digits. A key contribution of this study is the introduction of the "LMs with tutor" method, which meticulously guides the models through each computational step, similar to operations in multiple tape Turing machines. This method significantly improves the models' accuracy with out-of-distribution scenarios and repeated symbols, paving the way for future enhancements in symbolic manipulation tasks. Our project successfully trained the T5 and T5 random model to achieve addition results consistent with those in the paper. We initially tried enhancing the model with positional embeddings and later incorporated random positional embeddings for training up to 5 digits. This led to improvements with random positional embeddings for training up to 5 digits, though the out-of-distribution accuracy remains low.
  
</ul> 
<h2 dir="ltr">3.2 Chosen Result</h2> 
<ul dir="ltr"> 
We aimed to reproduce the effectiveness of positional markers in enhancing the performance of language models, specifically the T5 model, in arithmetic addition tasks. This is significant as it aligns with the main contribution of the original paper, which demonstrated how methods like detailed computation steps, callable functions, and positional markers can improve model performance in symbolic manipulation tasks. Your work particularly focused on the impact of positional markers, showing that they slightly outperformed the baseline model, especially in in-distribution data. This aligns with the findings in the original study where similar enhancements were shown to improve accuracy.

</ul>
<p><strong>Author's Motivation</strong><p>
<img width="300" alt="Screenshot 2024-05-15 at 11 36 29 PM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/8d239b45-f046-4756-b8e1-eae98803aaa4">
<p><strong>What We Will Focus On</strong><p>
<img width="341" alt="Screenshot 2024-05-15 at 11 49 25 PM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/4d9dab3a-de0f-42b7-8e8e-10a3c9c5629d">


<p><strong>What The Paper Accomplished</strong> </p>
<img width="314" alt="Screenshot 2024-05-15 at 11 38 42 PM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/a657d0a4-32ec-41c6-95e8-9fbb4c0d6744">


  
</ul> 
<h2 dir="ltr">3.3 Re-implementation Details</h2> 
<ul dir="ltr"> 
  <li>To reimplement the authors' paper, we needed to create a dataset and fine-tune a large language model (LLM). We chose to replicate the addition with positional markers result and decided to fine-tune the T5 small model because it was the smallest and required less computational power. Since the T5 model is not trained for addition, we introduced a special "add" token. Additionally, we had to create a baseline dataset, as the base T5 model cannot perform addition.

To create both datasets—the baseline and the positional markers baseline—we developed a function that takes in the digits (k), alpha value, and the number of samples. This function returns datasets distributed up to k digits with the specified number of samples. As presented by the authors of the paper, k, alpha, and in/out of distribution refer to the following:

"In-distribution refers to training on up to k-digit numbers and testing on up to k-digit numbers, while out-of-distribution refers to training on up to k-digit numbers and testing on numbers with more digits. α indicates the repetition level of the examples. An example x1 · · · xn with n digits is sampled with the next digit probability p(xi+1|xi) = α, when xi+1 = xi; otherwise, (1 − α)/9. Larger α indicates a higher repetition level."

The data then had to be formatted correctly to pass into the T5 model. This involved prefixing with the special "add" token, splitting the data into test, validation, and training sets, converting the data into a dataset object, and tokenizing it. The next step was to import the T5 model from the transformers library, along with the trainer. The training parameters specified by the paper included a learning rate of 5e-5, a batch size of 16, and 200 training epochs. We set the maximum generation length to 512 but decided to reduce it, as 512 seemed too long. Checkpoints were evaluated every 1000 optimization steps.

Once the setup was complete, the model was trained and saved to Google Drive or locally. To evaluate the model, we needed to assess both the baseline and positional markers performance. We created a dataset class of addition problems up to 30 digits long, with each class consisting of 100 randomly generated addition problems. These were then evaluated on the models and graphed. We chose 30 digits because that is what the authors of the paper used. The first 5 digits represent the in-distribution data, and 6-30 represent the out-of-distribution data.</li> 
    <p><strong>Required Libraries:</strong></p>
    <ul>
        <li>transformers</li>
        <li>tensorflow</li>
        <li>sklearn</li>
        <li>torch</li>
        <li>random</li>
    </ul>
    <p><strong>Instructions:</strong></p>
    <ol>
    <li>
        <strong>Create the Baseline and Positional Markers Dataset:</strong>
        <ul>
            <li>The necessary files are located in the <code>/data</code> directory.</li>
            <li>Adjust the <code>k</code>, <code>alpha</code>, and sample size parameters as needed.</li>
            <li>Note where the model will be saved. We saved it to Google Drive, but it can also be saved locally.</li>
        </ul>
    </li>
    <li>
        <strong>Train the Models:</strong>
        <ul>
            <li>Training instructions and files are available in the <code>/code</code> directory.</li>
            <li>Specify the dataset file path before training the model.</li>
            <li>After specifying the file path, you can train the model.</li>
            <li>Again, specify where to save the model. This can be done either locally or to Google Drive.</li>
        </ul>
    </li>
    <li>
        <strong>Examine the Results:</strong>
        <ul>
            <li>Files for examining the results are also found in the <code>/code</code> directory.</li>
            <li>The files labeled <code>generatePlot</code> will create addition examples for both in-distribution and out-of-distribution data.</li>
            <li>You can specify the number of digits and the number of problems per digit class.</li>
            <li>Ensure you specify the model name and its location.</li>
        </ul>
    </li>
    </ol>

    

  <p><strong>Computation Requirements:</strong></p>
  <ul>
    <li>T4 GPU 6 hours of training</li>
  </ul>
</ul> 
<h2 dir="ltr">3.4 Results and Analysis</h2> 
<ul dir="ltr"> 
  Our result performs well up to 3 digits (~50% accuracy) and we have trained on 200 epochs for each model (T5 and T5 random)
</ul>
<p></p>
<p dir="ltr"><strong>Result with regular baseline (Accuracy vs Class)</strong></p>
<div style="display: flex; justify-content: space-between;">
  <img width="508" alt="Screenshot 2024-05-16 at 12 16 16 AM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/91fbc6b3-87c9-4e7a-8c1f-8877af666470">
</ul>

  <img width="494" alt="Screenshot 2024-05-16 at 12 16 54 AM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/a948bfe4-70d3-4f68-994f-399198284377">
</div>
<p></p>
<p dir="ltr"><strong>Results with Positional Variable (Accuracy vs Class)</strong></p>
<div style="display: flex; justify-content: space-between;">
  <img width="508" alt="Screenshot 2024-05-16 at 12 17 27 AM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/e57a0928-e989-4154-a59f-62ea68e6ba59">
  <img width="494" alt="Screenshot 2024-05-16 at 12 16 54 AM" src="https://github.com/aad235/CS4782-Final-Project-LLMs-and-Arithmetic/assets/92837686/a948bfe4-70d3-4f68-994f-399198284377">
</div>
  
The model with Positional Markers slightly outperform baseline model. 

</ul> 
<h2 dir="ltr">3.5 Conclusion and Future Work</h2> 
<ul dir="ltr"> 
  <li>Summarize the key takeaways from your re-implementation effort and the lessons learned.</li> 
  <li>Discuss potential future directions or extensions based on your findings and the paper’s implications.</li> 
</ul> 
<h2 dir="ltr">3.6 References</h2> 
<p dir="ltr">• Include a list of references, including the original paper and any additional resources used in your re-implementation.</p> 
