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
  <li>Describe your re-implementation approach, including the model architec- ture, datasets, evaluation metrics, and any modifications made to the
    <ul dir="ltr"> 
      <li>riginal methodology.</li> 
    </ul></li> 
  <li>Provide instructions for running your code, including any dependencies, required libraries, and command-line arguments.</li> 
  <li>Specify the computational resources (e.g., GPU) needed to reproduce your results.</li> 
</ul> 
<h2 dir="ltr">3.4 Results and Analysis</h2> 
<ul dir="ltr"> 
  <li>Present your re-implementation results and compare them to the original paper’s findings.</li> 
  <li>Discuss any discrepancies or challenges encountered during the re-implementation process. If there is discrepancy between your results and the reported results, present a clear discussion of your hypotheses explaining the dis- crepancy.</li> 
  <li>Provide an analysis of your results in the context of the paper’s main contribution(s) and the broader research area.</li> 
  <li>Note: We are looking for a reasonable re-implementation of the method and a clear discussion of your results. A failure to match the reported results could happen for any number of reasons that would not negatively impact your grade. It is acceptable, for instance, to run smaller-scale experiments if you initially under-estimated the required resources for your selected result. It’s also possible that the authors left out some detail that is necessary to match their performance.</li> 
</ul> 
<h2 dir="ltr">3.5 Conclusion and Future Work</h2> 
<ul dir="ltr"> 
  <li>Summarize the key takeaways from your re-implementation effort and the lessons learned.</li> 
  <li>Discuss potential future directions or extensions based on your findings and the paper’s implications.</li> 
</ul> 
<h2 dir="ltr">3.6 References</h2> 
<p dir="ltr">• Include a list of references, including the original paper and any additional resources used in your re-implementation.</p> 
