## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers. 

**1. What is your project title?** 

Author Style Change Detection For Plagiarism Control Using Recurrent Neural Networks

**2. What is the problem that you want to solve?** 

Plagiarism remains a persistent challenge that requires the development of computationally efficient methods. Current popular systems, such as Turnitin, compare submitted documents against an archive of internet documents and media. New advances in NLP and deep learning allow for algorithms that detect changes within bodies of text. Style change detection is the only means to detect plagiarism in a document if no comparison text is available. The 2022 PAN challenge calls on researchers to identify text positions within a given multi-author document at which the author switches. We seek to understand if it is possible to find evidence for multiple authors writing a text together: do we have a means to detect variations in the writing style?

**3. What deep learning methodologies do you plan to use in your project?** 

- We can frame this as a classification task by looking at pairs of paragraphs (two features: 1st row => 1st paragraph vs. 2nd paragraph, 2nd row => 2nd paragraph vs. 3rd paragraph, etc.), with the output variable being ‘same’ or ‘different’ author.
- We can follow the lead of Nath (2021) we can create word embeddings and then set up Siamese NNs with a bidirectional LTSM or GRU layer. There are a few opportunities for us to build on the method in Nath (2021):
  - We could use different embedding methods, like BERT and compare in terms of performance on this kind of corpus. 
  - We could use an attention based layer
  - We could experiment with initialization, drop-out, different activation functions, etc.

**4. What dataset will you use? Provide information about the dataset, and a url for the dataset if available. Briefly discuss suitability of the dataset for your problem.** 

The dataset is based on forum posts from StackExchange. It consists of a training set of 11,200 documents and validation set of 2,400 documents. The corpus is large enough to train neural networks, but small enough in order to train models within reasonable time. 

*Example features of one document:* Paragraphs. `[par1_aut1, par2_aut1, par3_aut2, par4_aut2, par5_aut2,...]`

*Example outcome vector of one document:* Style author changes `[0,0,1,0,0,...]`

Link to competition: https://pan.webis.de/clef22/pan22-web/style-change-detection.html

Link to dataset: https://zenodo.org/record/6334245

**5. List key references (e.g. research papers) that your project will be based on?** 

Nath, Sukanya (2021). *Style change detection using Siamese neural networks*, Notebook for PAN at CLEF 2021. Accessed 22 March 2022: http://ceur-ws.org/Vol-2936/paper-183.pdf

Summary paper from last year: https://pan.webis.de/publications.html?q=2021+Zangerle

**Please indicate whether your project proposal is ready for review (Yes/No):** Yes.

## Feedback

**[MV, 24 March 2022]** Project proposal approved. The problem sounds interesting. You have identified a dataset and a set of references to base your work on. I trust that you will explore different neural network architectures, and may consider different training methods. The proposed problem is formulated as a binary classification problem -- for a pair of paragraphs asking whether they were written by the same author. This sounds good. This problem formulation may be compared with a multi-class classification problem formulation - for each paragraph predict the identity of the author. You may also think about other problems that can be formulated in the given setting. For example, Task 1 in PAN22 is a change point detection problem for a time series of paragraphs. Change point detection is a standard problem studied in statistics (for a time series of numerical vectors), so you may draw some inspiration from there.  
